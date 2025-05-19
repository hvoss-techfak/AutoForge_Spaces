import string
import uuid
import os
import logging
import zipfile

import sentry_sdk
import wandb
from sentry_sdk import capture_exception
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration
import spaces
dsn = os.getenv("SENTRY_DSN")
if not dsn:
    print("WARNING: SENTRY_DSN not set – Sentry disabled")
else:

    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=0.1,
        integrations=[
            StarletteIntegration(
                failed_request_status_codes={
                    400,
                    422,
                    *range(500, 599),
                },  # also log 4xx from Gradio
            ),
            LoggingIntegration(
                level=logging.INFO,  # breadcrumb level
                event_level=logging.ERROR,
            ),
            FastApiIntegration(),
        ],
        release=os.getenv("HF_SPACE_VERSION", "dev"),
        environment="hf_space",
    )

sentry_sdk.capture_message("🎉 Sentry is wired up!")

USE_WANDB = "WANDB_API_KEY" in os.environ
if USE_WANDB:
    wandb.login(key=os.environ["WANDB_API_KEY"])

else:
    print("Warning: WANDB_API_KEY not set. Skipping wandb logging.")



import gradio
import functools
from sentry_sdk import flush

orig_call_fn = gradio.blocks.Blocks.call_function  # present in all 3.x & 4.x

@functools.wraps(orig_call_fn)
async def sentry_call_fn(self, *args, **kwargs):
    try:
        return await orig_call_fn(self, *args, **kwargs)
    except Exception as exc:
        capture_exception(exc)
        flush(timeout=2)
        raise

gradio.blocks.Blocks.call_function = sentry_call_fn


import gradio as gr
import pandas as pd
import os
import subprocess
import time
import sys
from datetime import datetime
import re

# --- Configuration ---
#AUTFORGE_SCRIPT_PATH = "auto_forge.py"  # Make sure this points to your script
DEFAULT_MATERIALS_CSV = "default_materials.csv"
GRADIO_OUTPUT_BASE_DIR = "output"
os.makedirs(GRADIO_OUTPUT_BASE_DIR, exist_ok=True)

REQUIRED_SCRIPT_COLS = ["Brand", " Name", " TD", " Color"]
DISPLAY_COL_MAP = {
    "Brand": "Brand",
    " Name": "Name",
    " TD": "TD",
    " Color": "Color (Hex)",
}


def _check_quota(required_sec: int):
    """
    Check if the user has enough ZeroGPU quota remaining.
    Raises RuntimeError if not enough.
    """
    remaining = int(os.getenv("ZEROGPU_REMAINING", "0"))
    if remaining < required_sec:
        raise RuntimeError(
            f"Insufficient ZeroGPU quota: need {required_sec}s but only {remaining}s left.\n"
            "Please log in to Hugging Face or wait a few minutes for quota to recharge."
        )


def ensure_required_cols(df, *, in_display_space):
    """
    Return a copy of *df* with every required column present.
    If *in_display_space* is True we use the display names
    (Brand, Name, TD, Color (Hex)); otherwise we use the script names.
    """
    target_cols = (
        DISPLAY_COL_MAP if in_display_space else {k: k for k in REQUIRED_SCRIPT_COLS}
    )
    df_fixed = df.copy()
    for col_script, col_display in target_cols.items():
        if col_display not in df_fixed.columns:
            # sensible defaults
            if "TD" in col_display:
                default = 0.0
            elif "Color" in col_display:
                default = "#000000"
            elif "Owned" in col_display:  # NEW
                default = "false"
            else:
                default = ""
            df_fixed[col_display] = default
    # order columns nicely
    return df_fixed[list(target_cols.values())]


def rgba_to_hex(col: str) -> str:
    """
    Turn 'rgba(r, g, b, a)' or 'rgb(r, g, b)' into '#RRGGBB'.
    If the input is already a hex code or anything unexpected,
    return it unchanged.
    """
    if not isinstance(col, str):
        return col
    col = col.strip()
    if col.startswith("#"):          # already fine
        return col.upper()

    m = re.match(
        r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*[\d.]+)?\s*\)",
        col,
    )
    if not m:
        return col                   # not something we recognise

    r, g, b = (int(float(x)) for x in m.groups()[:3])
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def zip_dir_no_compress(src_dir: str, dest_zip: str) -> str:
    """Create *dest_zip* from *src_dir* using no compression (ZIP_STORED)."""
    t0 = time.time()
    with zipfile.ZipFile(dest_zip, "w",
                         compression=zipfile.ZIP_STORED,
                         allowZip64=True) as zf:
        for root, _, files in os.walk(src_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                # keep folder structure inside the archive but drop the leading path
                zf.write(fpath, os.path.relpath(fpath, src_dir))
    print(f"Zipping finished in {time.time() - t0:.1f}s")
    return dest_zip

# --- Helper Functions ---
def get_script_args_info(exclude_args=None):
    if exclude_args is None:
        exclude_args = []

    all_args_info = [
        # input_image is handled separately in the UI
        {
            "name": "--iterations",
            "type": "number",
            "default": 2000,
            "help": "Number of optimization iterations",
        },
        {
            "name": "--layer_height",
            "type": "number",
            "default": 0.04,
            "step": 0.01,
            "help": "Layer thickness in mm",
        },
        {
            "name": "--max_layers",
            "type": "number",
            "default": 75,
            "precision": 0,
            "help": "Maximum number of layers",
        },
        {
            "name": "--learning_rate",
            "type": "number",
            "default": 0.015,
            "step": 0.001,
            "help": "Learning rate for optimization",
        },
        {
            "name": "--background_height",
            "type": "number",
            "default": 0.4,
            "step": 0.01,
            "help": "Height of the background in mm",
        },
        {
            "name": "--background_color",
            "type": "colorpicker",
            "default": "#000000",
            "help": "Background color",
        },
        {
            "name": "--stl_output_size",
            "type": "number",
            "default": 100,
            "precision": 0,
            "help": "Size of the longest dimension of the output STL file in mm",
        },
        {
            "name": "--nozzle_diameter",
            "type": "number",
            "default": 0.4,
            "step": 0.1,
            "help": "Diameter of the printer nozzle in mm",
        },
        {
            "name": "--pruning_max_colors",
            "type": "number",
            "default": 10,
            "precision": 0,
            "help": "Max number of colors allowed after pruning",
        },
        {
            "name": "--pruning_max_swaps",
            "type": "number",
            "default": 20,
            "precision": 0,
            "help": "Max number of swaps allowed after pruning",
        },
        {
            "name": "--pruning_max_layer",
            "type": "number",
            "default": 75,
            "precision": 0,
            "help": "Max number of layers allowed after pruning",
        },
        {
            "name": "--warmup_fraction",
            "type": "slider",
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Fraction of iterations for keeping the tau at the initial value",
        },
        {
            "name": "--learning_rate_warmup_fraction",
            "type": "slider",
            "default": 0.25,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Fraction of iterations that the learning rate is increasing (warmup)",
        },
        # {
        #     "name": "--init_tau",
        #     "type": "number",
        #     "default": 1.0,
        #     "help": "Initial tau value for Gumbel-Softmax",
        # },
        # {
        #     "name": "--final_tau",
        #     "type": "number",
        #     "default": 0.01,
        #     "help": "Final tau value for Gumbel-Softmax",
        # },
        # {
        #     "name": "--min_layers",
        #     "type": "number",
        #     "default": 0,
        #     "precision": 0,
        #     "help": "Minimum number of layers. Used for pruning.",
        # },
        {
            "name": "--early_stopping",
            "type": "number",
            "default": 1500,
            "precision": 0,
            "help": "Number of steps without improvement before stopping",
        },
        {
            "name": "--random_seed",
            "type": "number",
            "default": 0,
            "precision": 0,
            "help": "Specify the random seed, or use 0 for automatic generation",
        },
        {
            "name": "--num_init_rounds",
            "type": "number",
            "default": 64,
            "precision": 0,
            "help": "Number of rounds to choose the starting height map from.",
        },
    ]
    return [arg for arg in all_args_info if arg["name"] not in exclude_args]


# Initial filament data
initial_filament_data = {
    "Brand": ["Generic", "Generic", "Generic","Generic","Generic","Generic",],
    " Name": ["PLA Black", "PLA Grey", "PLA White","PLA Red","PLA Green","PLA Blue"],
    " TD": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    " Color": ["#000000", "#808080", "#FFFFFF","#FF0000","#00FF00","#0000FF"],
    " Owned": ["true", "true", "true", "true", "true", "true"],  # ← add
}
initial_df = pd.DataFrame(initial_filament_data)

if os.path.exists(DEFAULT_MATERIALS_CSV):
    try:
        initial_df = pd.read_csv(DEFAULT_MATERIALS_CSV)
        for col in ["Brand", " Name", " TD", " Color"]:
            if col not in initial_df.columns:
                initial_df[col] = None
        initial_df = initial_df[["Brand", " Name", " TD", " Color"]].astype(
            {" TD": float, " Color": str}
        )
    except Exception as e:
        print(f"Warning: Could not load {DEFAULT_MATERIALS_CSV}: {e}. Using default.")
        initial_df = pd.DataFrame(initial_filament_data)
else:
    initial_df.to_csv(DEFAULT_MATERIALS_CSV, index=False)

@spaces.GPU(duration=90)
def run_autoforge_process(cmd, log_path):
    """Run AutoForge in-process and stream its console output to *log_path*."""
    _check_quota(90)

    cli_args = cmd[1:]          # skip the literal "autoforge"
    autoforge_main = importlib.import_module("autoforge.__main__")

    exit_code = 0
    with open(log_path, "w", buffering=1, encoding="utf-8") as log_f, \
         redirect_stdout(log_f), redirect_stderr(log_f):
        try:
            sys.argv = ["autoforge"] + cli_args
            autoforge_main.main()         # runs until completion
        except SystemExit as e:           # AutoForge calls sys.exit()
            exit_code = e.code

    return exit_code


# Helper for creating an empty 10-tuple for error returns
def create_empty_error_outputs(log_message=""):
    return (
        log_message,  # progress_output
        None,  # final_image_preview
        gr.update(visible=False, interactive=False),  # ### ZIP: download_zip
    )


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# [Autoforge](https://github.com/hvoss-techfak/AutoForge) Web UI")

    filament_df_state = gr.State(initial_df.copy())
    current_run_output_dir = gr.State(None)

    with gr.Tabs():
        with gr.TabItem("Filament Management"):
            gr.Markdown(
                'Manage your filament list here. This list will be used by Autoforge during the optimization process.'
            )
            gr.Markdown(
                'If you have Hueforge, you can export your filaments under "Filaments -> Export" in the Hueforge software. Please make sure to select "CSV" instead of "JSON" during the export dialog.'
            )
            gr.Markdown(
                'To remove a filament simply right-click on any of the fields and select "Delete Row"'
            )
            with gr.Row():
                load_csv_button = gr.UploadButton(
                    "Load Filaments CSV", file_types=[".csv"]
                )
                save_csv_button = gr.Button("Save Current Filaments to CSV")
            filament_table = gr.DataFrame(
                value=ensure_required_cols(
                    initial_df.copy().rename(
                        columns={" Name": "Name", " TD": "TD", " Color": "Color (Hex)"}
                    ),
                    in_display_space=True,
                ),
                headers=["Brand", "Name", "TD", "Color (Hex)"],
                datatype=["str", "str", "number", "str"],
                interactive=True,
                label="Filaments",
            )
            gr.Markdown("## Add New Filament")
            with gr.Row():
                new_brand = gr.Textbox(label="Brand")
                new_name = gr.Textbox(label="Name")
            with gr.Row():
                new_td = gr.Number(
                    label="TD (Transmission/Opacity)",
                    value=1.0,
                    minimum=0,
                    maximum=100,
                    step=0.1,
                )
                new_color_hex = gr.ColorPicker(label="Color", value="#FF0000")
            add_filament_button = gr.Button("Add Filament to Table")
            download_csv_trigger = gr.File(
                label="Download Filament CSV", visible=False, interactive=False
            )

            def update_filament_df_state_from_table(display_df):
                display_df = ensure_required_cols(display_df, in_display_space=True)

                # make sure every colour is hex
                if "Color (Hex)" in display_df.columns:
                    display_df["Color (Hex)"] = display_df["Color (Hex)"].apply(
                        rgba_to_hex
                    )

                script_df = display_df.rename(
                    columns={"Name": " Name", "TD": " TD", "Color (Hex)": " Color"}
                )
                script_df = ensure_required_cols(script_df, in_display_space=False)
                filament_df_state.value = script_df

            def add_filament_to_table(current_display_df, brand, name, td, color_hex):
                if not brand or not name:
                    gr.Warning("Brand and Name cannot be empty.")
                    return current_display_df

                color_hex = rgba_to_hex(color_hex)  # <-- new line

                new_row = pd.DataFrame(
                    [{"Brand": brand, "Name": name, "TD": td, "Color (Hex)": color_hex}]
                )
                updated_display_df = pd.concat(
                    [current_display_df, new_row], ignore_index=True
                )
                update_filament_df_state_from_table(updated_display_df)
                return updated_display_df

            def load_filaments_from_csv_upload(file_obj):
                if file_obj is None:
                    current_script_df = filament_df_state.value
                    if current_script_df is not None and not current_script_df.empty:
                        return current_script_df.rename(
                            columns={
                                " Name": "Name",
                                " TD": "TD",
                                " Color": "Color (Hex)",
                            }
                        )
                    return initial_df.copy().rename(
                        columns={" Name": "Name", " TD": "TD", " Color": "Color (Hex)"}
                    )
                try:
                    loaded_script_df = pd.read_csv(file_obj.name)
                    loaded_script_df = ensure_required_cols(
                        loaded_script_df, in_display_space=False
                    )
                    expected_cols = ["Brand", " Name", " TD", " Color"]
                    if not all(
                        col in loaded_script_df.columns for col in expected_cols
                    ):
                        gr.Error(
                            f"CSV must contain columns: {', '.join(expected_cols)}. Found: {loaded_script_df.columns.tolist()}"
                        )
                        capture_exception(
                            Exception(
                                f"CSV must contain columns: {', '.join(expected_cols)}. Found: {loaded_script_df.columns.tolist()}"
                            )
                        )
                        current_script_df = filament_df_state.value
                        if (
                            current_script_df is not None
                            and not current_script_df.empty
                        ):
                            return current_script_df.rename(
                                columns={
                                    " Name": "Name",
                                    " TD": "TD",
                                    " Color": "Color (Hex)",
                                }
                            )
                        return initial_df.copy().rename(
                            columns={
                                " Name": "Name",
                                " TD": "TD",
                                " Color": "Color (Hex)",
                            }
                        )
                    filament_df_state.value = loaded_script_df.copy()
                    return loaded_script_df.rename(
                        columns={" Name": "Name", " TD": "TD", " Color": "Color (Hex)"}
                    )
                except Exception as e:
                    gr.Error(f"Error loading CSV: {e}")
                    capture_exception(e)
                    current_script_df = filament_df_state.value
                    if current_script_df is not None and not current_script_df.empty:
                        return current_script_df.rename(
                            columns={
                                " Name": "Name",
                                " TD": "TD",
                                " Color": "Color (Hex)",
                            }
                        )
                    return initial_df.copy().rename(
                        columns={" Name": "Name", " TD": "TD", " Color": "Color (Hex)"}
                    )

            def save_filaments_to_file_for_download(current_script_df_from_state):
                if (
                    current_script_df_from_state is None
                    or current_script_df_from_state.empty
                ):
                    gr.Warning("Filament table is empty. Nothing to save.")
                    return None
                df_to_save = current_script_df_from_state.copy()
                required_cols = ["Brand", " Name", " TD", " Color"]
                if not all(col in df_to_save.columns for col in required_cols):
                    gr.Error(
                        f"Cannot save. DataFrame missing required script columns. Expected: {required_cols}. Found: {df_to_save.columns.tolist()}"
                    )
                    capture_exception(Exception(f"Missing columns: {df_to_save.columns.tolist()}"))
                    return None
                temp_dir = os.path.join(GRADIO_OUTPUT_BASE_DIR, "_temp_downloads")
                os.makedirs(temp_dir, exist_ok=True)
                temp_filament_csv_path = os.path.join(
                    temp_dir,
                    f"filaments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                )
                try:
                    df_to_save.to_csv(temp_filament_csv_path, index=False)
                    gr.Info("Filaments prepared for download.")
                    return gr.File(
                        value=temp_filament_csv_path,
                        label="Download Filament CSV",
                        interactive=True,
                        visible=True,
                    )
                except Exception as e:
                    capture_exception(e)
                    gr.Error(f"Error saving CSV for download: {e}")
                    return None

            filament_table.change(
                update_filament_df_state_from_table,
                inputs=[filament_table],
                outputs=None,
                queue=False,
            )
            add_filament_button.click(
                add_filament_to_table,
                inputs=[filament_table, new_brand, new_name, new_td, new_color_hex],
                outputs=[filament_table],
            )
            load_csv_button.upload(
                load_filaments_from_csv_upload,
                inputs=[load_csv_button],
                outputs=[filament_table],
            )
            save_csv_button.click(
                save_filaments_to_file_for_download,
                inputs=[filament_df_state],
                outputs=[download_csv_trigger],
            )

        with gr.TabItem("Run Autoforge"):

            accordion_params_dict = {}
            accordion_params_ordered_names = []

            gr.Markdown(
                'Here you can upload an image, adjust the parameters and run the Autoforge process. The filaments from the "Filament Management" Tab are automatically used. After the process completes you can download the results at the bottom of the page.'
            )
            gr.Markdown(
                'If you want to limit the number of colors or color swaps you can find the option under the "Autoforge Parameters" as "pruning_max_colors" and "pruning_max_swaps"'
            )
            gr.Markdown(
                'Please note that huggingface enforces a maximum execution time of one minute. Depending on your configuration (especially iteration count) it is possible to exceed this time limit. In that case you will see a "GPU Task aborted" error or simply "Error".'
                ' If you need more time, take a look at the [Autoforge Github Page](https://github.com/hvoss-techfak/AutoForge) to see how you can run the program locally, or pull the docker container for this project (upper right corner -> three dots -> "run locally")'
            )
            gr.Markdown(
                'Hint: If you want to improve the quality of the output, try increasing "iterations" to 4000, and "num_init_rounds" to 256, but be warned that this can lead to out-of-time errors due to the time restrictions mentioned above.'
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input Image (Required)")
                    input_image_component = gr.Image(         # keep transparency alive
                        type="pil",            # <- no temporary JPEG cache
                        image_mode="RGBA",     # tells Gradio to expect alpha
                        label="Upload Image",
                        sources=["upload"],
                        interactive=True,
                    )
                with gr.Column(scale=2):
                    gr.Markdown("### Preview")
                    with gr.Accordion("Progress & Output", open=True):
                        final_image_preview = gr.Image(
                            label="Model Preview",
                            type="filepath",
                            interactive=False,
                        )

            with gr.Row():
                with gr.Accordion("Autoforge Parameters", open=False):
                    args_for_accordion = get_script_args_info(
                        exclude_args=["--input_image"]
                    )

                    for arg in args_for_accordion:
                        label, info, default_val = (
                            f"{arg['name']}",
                            arg["help"],
                            arg.get("default"),
                        )
                        if arg["type"] == "number":
                            accordion_params_dict[arg["name"]] = gr.Number(
                                label=label,
                                value=default_val,
                                info=info,
                                minimum=arg.get("min"),
                                maximum=arg.get("max"),
                                step=arg.get(
                                    "step",
                                    0.001 if isinstance(default_val, float) else 1,
                                ),
                                precision=arg.get("precision", None),
                            )
                        elif arg["type"] == "slider":
                            accordion_params_dict[arg["name"]] = gr.Slider(
                                label=label,
                                value=default_val,
                                info=info,
                                minimum=arg.get("min", 0),
                                maximum=arg.get("max", 1),
                                step=arg.get("step", 0.01),
                            )
                        elif arg["type"] == "checkbox":
                            accordion_params_dict[arg["name"]] = gr.Checkbox(
                                label=label, value=default_val, info=info
                            )
                        elif arg["type"] == "colorpicker":
                            accordion_params_dict[arg["name"]] = gr.ColorPicker(
                                label=label, value=default_val, info=info
                            )
                        else:
                            accordion_params_dict[arg["name"]] = gr.Textbox(
                                label=label, value=str(default_val), info=info
                            )
                        accordion_params_ordered_names.append(arg["name"])

            run_button = gr.Button(
                "Run Autoforge Process",
                variant="primary",
                elem_id="run_button_full_width",
            )


            progress_output = gr.Textbox(
                label="Console Output",
                lines=15,
                autoscroll=True,
                show_copy_button=False,
            )

            with gr.Row():
                download_results = gr.File(
                    label="Download results",
                    file_count="multiple",
                    interactive=True,
                    visible=False,
                )

    # --- Backend Function for Running the Script ---
    def execute_autoforge_script(
        current_filaments_df_state_val, input_image, *accordion_param_values
    ):

        # 0. Validate Inputs
        if input_image is None:
            gr.Error("Input Image is required! Please upload an image.")
            capture_exception(Exception("Input Image is required!"))
            return create_empty_error_outputs("Error: Input Image is required!")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())
        run_output_dir_val = os.path.join(GRADIO_OUTPUT_BASE_DIR, f"run_{timestamp}")
        os.makedirs(run_output_dir_val, exist_ok=True)
        current_run_output_dir.value = run_output_dir_val

        # 1. Save current filaments
        if (
            current_filaments_df_state_val is None
            or current_filaments_df_state_val.empty
        ):
            gr.Error("Filament table is empty. Please add filaments.")
            capture_exception(
                Exception("Filament table is empty. Please add filaments.")
            )
            return create_empty_error_outputs("Error: Filament table is empty.")

        temp_filament_csv = os.path.join(run_output_dir_val, "materials.csv")
        df_to_save = current_filaments_df_state_val.copy()
        required_cols = ["Brand", " Name", " TD", " Color"]
        missing_cols = [col for col in required_cols if col not in df_to_save.columns]
        if missing_cols:
            err_msg = (
                f"Error: Filament data is missing columns: {', '.join(missing_cols)}."
            )
            gr.Error(err_msg)
            capture_exception(
                Exception(f"Filament data is missing columns: {', '.join(missing_cols)}.")
            )
            return create_empty_error_outputs(err_msg)
        try:
            df_to_save.to_csv(temp_filament_csv, index=False)
        except Exception as e:
            capture_exception(e)
            err_msg = f"Error saving temporary filament CSV: {e}"
            gr.Error(err_msg)
            return create_empty_error_outputs(err_msg)

        # 2. Construct command
        python_executable = sys.executable or "python"
        command = ["autoforge",]
        command.extend(["--csv_file", temp_filament_csv])
        command.extend(["--output_folder", run_output_dir_val])
        command.extend(["--disable_visualization_for_gradio","1"])

        try:
            # decide where to store the image we pass to Autoforge
            script_input_image_path = os.path.join(
                run_output_dir_val, "input_image.png"
            )
            input_image.save(script_input_image_path, format="PNG")
            command.extend(["--input_image", script_input_image_path])
        except Exception as e:
            capture_exception(e)
            err_msg = f"Error handling input image: {e}"
            gr.Error(err_msg)
            return create_empty_error_outputs(err_msg)

        param_dict = dict(zip(accordion_params_ordered_names, accordion_param_values))
        for arg_name, arg_widget_val in param_dict.items():
            if arg_widget_val is None or arg_widget_val == "":
                arg_info_list = [
                    item for item in get_script_args_info() if item["name"] == arg_name
                ]  # get full list to check type
                if (
                    arg_info_list
                    and arg_info_list[0]["type"] == "checkbox"
                    and arg_widget_val is False
                ):
                    continue
                else:
                    continue

            if arg_name == "--background_color":
                arg_widget_val = rgba_to_hex(arg_widget_val)

            if isinstance(arg_widget_val, bool):
                if arg_widget_val:
                    command.append(arg_name)
            else:
                command.extend([arg_name, str(arg_widget_val)])


        # 3. Run script
        log_output = [
            "Starting Autoforge process at ",
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"Output directory: {run_output_dir_val}\n",
            f"Command: {' '.join(command)}\n\n",
        ]

        yield create_empty_error_outputs(log_output)  # clear UI and show header


        def _maybe_new_preview():
            """
            If vis_temp.png has a newer mtime than last time, copy it to a
            stamped name (to defeat browser cache) and return that path.
            Otherwise return gr.update() so the image stays as-is.
            """
            from gradio import update  # local import for clarity

            nonlocal preview_mtime

            src = os.path.join(run_output_dir_val, "vis_temp.png")
            if not os.path.exists(src):
                return update()  # nothing new, keep old

            mtime = os.path.getmtime(src)
            if mtime <= preview_mtime:  # unchanged
                return update()  # → no UI update

            return src  # → refresh image

        # ---- run Autoforge on the GPU in a helper thread ------------------

        log_file = os.path.join(run_output_dir_val, "autoforge_live.log")
        open(log_file, "w", encoding="utf-8").close()

        cmd_str = " ".join(command)
        sentry_sdk.capture_event(
            {
                "message": "Autoforge process started",
                "level": "info",
                "fingerprint": ["autoforge-process-start"],  # every start groups here
                "extra": {"command": cmd_str},  # still searchable
            }
        )

        # simple thread that just calls the GPU helper and stores the exit code
        import threading

        class Worker(threading.Thread):
            def __init__(self, cmd, log_path):
                super().__init__(daemon=True)
                self.cmd, self.log_path = cmd, log_path
                self.returncode = None

            def run(self):
                try:
                    self.returncode = run_autoforge_process(self.cmd, self.log_path)
                except Exception as e:
                    self.exc = e
                    capture_exception(e)  # still goes to Sentry
                    import traceback
                    exc_str = "".join(traceback.format_exception_only(e)).strip()
                    # make the error visible in the UI console
                    with open(self.log_path, "a", encoding="utf-8") as lf:
                        lf.write(
                            "\nERROR: {}. This usually means that you or the space has no free GPU "
                            "minutes left, or the process took too long due to too many filaments or changed parameters. Please clone the docker container, run it locally or wait for a bit.\n".format(exc_str)
                        )
                    gr.Error(
                        "ERROR: {}. This usually means that you ore the the space has no free GPU "
                        "minutes left, or the process took too long due to too many filaments or changed parameters. Please clone the docker container, run it locally or wait for a bit.\n".format(exc_str)
                    )
                    # a non-zero code tells the outer loop something went wrong
                    self.returncode = -1

        try:
            worker = Worker(command, log_file)
            worker.start()
    
            preview_mtime = 0
            last_push = 0
            file_pos = 0  # how far we've read
    
            while worker.is_alive() or file_pos < os.path.getsize(log_file):
                # read any new console text
                with open(log_file, "r", encoding="utf-8") as lf:
                    lf.seek(file_pos)
                    new_txt = lf.read()
                    file_pos = lf.tell()
                    log_output += new_txt
    
                now = time.time()
                if now - last_push >= 1.0:  # one-second UI tick
                    current_preview = _maybe_new_preview()
                    yield (
                        "".join(log_output),
                        current_preview,
                        gr.update(),  # placeholder for download widget
                    )
                    last_push = now
    
                time.sleep(0.05)
    
            worker.join()  # make sure it’s done
        except RuntimeError as e:
            # Show toast to user
            log_output += str(e)
            gr.Error(str(e))  # <-- this is the toast
            capture_exception(e)
            return create_empty_error_outputs(str(e))

        if getattr(worker, "exc", None) is not None:
            # worker.exc will be the ZeroGPU / scheduler error
            err_msg = f"GPU run failed: {worker.exc}"
            log_output += f"\n{err_msg}\n"
            gr.Error(err_msg)           # toast
            yield (                     # push the message into the textbox
                "".join(log_output),
                _maybe_new_preview(),
                gr.update(),
            )
            return                      # stop the coroutine cleanly

        # If the GPU scheduler threw, we already wrote the text into the log.
        # Just read the tail once more so it reaches the UI textbox.
        with open(log_file, "r", encoding="utf-8") as lf:
            lf.seek(file_pos)
            log_output += lf.read()

        return_code = worker.returncode

        try:
            sentry_sdk.add_attachment(
                path=log_file,
                filename="autoforge.log",
                content_type="text/plain",
            )
        except Exception as e:
            capture_exception(e)

        if worker.returncode != 0:
            err_msg = (
                f"Autoforge exited with code {worker.returncode}\n"
                "See the console output above for details."
            )
            log_output += f"\n{err_msg}\n"
            gr.Error(err_msg)
            yield (
                "".join(log_output),
                _maybe_new_preview(),
                gr.update(),
            )
            return
        log_output += (
            "\nAutoforge process completed successfully!"
            if return_code == 0
            else f"\nAutoforge process failed with exit code {return_code}."
        )
        log_str = " ".join(log_output)



        files_to_offer = [
            p
            for p in [
                os.path.join(run_output_dir_val, "final_model.png"),
                os.path.join(run_output_dir_val, "final_model.stl"),
                os.path.join(run_output_dir_val, "swap_instructions.txt"),
                os.path.join(run_output_dir_val, "project_file.hfp"),
            ]
            if os.path.exists(p)
        ]
        png_path = os.path.join(run_output_dir_val, "final_model.png")
        out_png = png_path if os.path.exists(png_path) else None

        if out_png is None:
            log_output += "\nWarning: final_model.png not found in output."

        sentry_sdk.capture_event(  # moved inside the same scope
            {
                "message": "Autoforge process finished",
                "level": "info",
                "fingerprint": ["autoforge-process-finished"],
                "extra": {"log": log_str},
            }
        )

        if USE_WANDB:
            run = None
            try:
                run = wandb.init(
                    project="autoforge",
                    name=f"run_{timestamp}",
                    notes="Autoforge Web UI run",
                    tags=["autoforge", "gradio"],
                )
                wlogs= {"input_image": wandb.Image(script_input_image_path),}
                if out_png:
                    wlogs["output_image"] = wandb.Image(out_png)

                material_csv = pd.read_csv(temp_filament_csv)
                table = wandb.Table(dataframe=material_csv)
                wlogs["materials"] = table
                #log log_output as pandas table
                from wandb import Html
                log_text = "".join(log_output).replace("\r", "\n")

                def clean_log_strict(text: str) -> str:
                    # Keep only printable characters + newline + tab
                    allowed = set(string.printable) | {"\n", "\t"}
                    return "".join(ch for ch in text if ch in allowed)

                log_text_cleaned = clean_log_strict(log_text)
                wlogs["log"] = Html(f"<pre>{log_text_cleaned}</pre>")


                wandb.log(wlogs)
            except Exception as e:
                #we don't want wandb errors logged in sentry
                print(e)
            finally:
                if run is not None:
                    run.finish()

        yield (
            "".join(log_output),  # progress_output
            out_png,  # final_image_preview (same as before)
            gr.update(  # download_results
                value=files_to_offer,
                visible=True,
                interactive=True,
            ),
        )

    run_inputs = [filament_df_state, input_image_component] + [
        accordion_params_dict[name] for name in accordion_params_ordered_names
    ]
    run_outputs = [
        progress_output,
        final_image_preview,
        download_results,  # ### ZIP PATCH: only three outputs now
    ]

    run_button.click(execute_autoforge_script, inputs=run_inputs, outputs=run_outputs)

css = """ #run_button_full_width { width: 100%; } """
if __name__ == "__main__":

    if not os.path.exists(DEFAULT_MATERIALS_CSV):
        print(f"Creating default filament file: {DEFAULT_MATERIALS_CSV}")
        try:
            initial_df.to_csv(DEFAULT_MATERIALS_CSV, index=False)
        except Exception as e:
            print(f"Could not write default {DEFAULT_MATERIALS_CSV}: {e}")
    print("To run the UI, execute: python app.py")  # Corrected to python app.py
    demo.queue(default_concurrency_limit=1).launch(share=False)
