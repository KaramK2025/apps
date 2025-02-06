import streamlit as st
import os
import fitz
import re
import pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from PIL import Image
import time
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from datetime import datetime

# Hidden base directory (not shown to user)
base_dir = r"Projects"

reader = easyocr.Reader(['en'])
st.set_page_config(layout="wide")

if "all_annotated_images" not in st.session_state:
    st.session_state.all_annotated_images = []
if "all_page_data" not in st.session_state:
    st.session_state.all_page_data = {}

model_path = "best.pt"
model = YOLO(model_path)

def convert_pdf_to_jpegs(pdf_path, output_dir, pages_to_ignore):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    scale = 4.0
    mat = fitz.Matrix(scale, scale)
    jpeg_paths = []
    for i in range(len(doc)):
        if (i + 1) in pages_to_ignore:
            continue
        page = doc[i]
        pix = page.get_pixmap(matrix=mat)
        out_file = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i+1}.jpg"
        out_path = os.path.join(output_dir, out_file)
        pix.save(out_path)
        jpeg_paths.append((i + 1, os.path.abspath(out_path)))
    return jpeg_paths

def extract_wl_number(text):
    match = re.search(r'WL\s*[:\-]*\s*([\d\.]+)', text, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def perform_ocr_with_easyocr(image):
    result = reader.readtext(image)
    extracted_text = " ".join([text[1] for text in result])
    return extracted_text.strip()

st.title("Callout Extractor")

with st.form("input_form"):
    project_description = st.text_input("Project Description")
    pdf_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    pages_to_ignore = st.text_input("Pages to Ignore (e.g., 1-3, 5, 7-9)", value="0")
    submitted = st.form_submit_button("Run Model")

if submitted:
    if not pdf_files or not project_description.strip():
        st.warning("Please provide a project description and upload at least one PDF.")
    else:
        # Create the project directory
        project_dir = os.path.join(base_dir, project_description)
        os.makedirs(project_dir, exist_ok=True)
        
        st.session_state.all_annotated_images.clear()
        st.session_state.all_page_data.clear()

        for pdf_file in pdf_files:
            pdf_filename = pdf_file.name
            wo_num = os.path.splitext(pdf_filename)[0]

            st.write(f"**Processing:** {pdf_filename} (WO#: {wo_num})")

            # Each PDF run directory under the project
            pdf_run_dir = os.path.join(project_dir, wo_num)
            jpegs_path = os.path.join(pdf_run_dir, "jpegs")
            os.makedirs(jpegs_path, exist_ok=True)

            # Parse pages to ignore
            pages_to_ignore_set = set()
            for part in pages_to_ignore.split(","):
                part = part.strip()
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    pages_to_ignore_set.update(range(start, end + 1))
                else:
                    if part.isdigit():
                        pages_to_ignore_set.add(int(part))

            # Save the PDF in its run directory
            pdf_export_path = os.path.join(pdf_run_dir, f"{wo_num}.pdf")
            with open(pdf_export_path, "wb") as f:
                f.write(pdf_file.read())

            start_time = time.time()
            progress_msg = st.empty()

            with st.spinner("Converting PDF..."):
                jpeg_list = convert_pdf_to_jpegs(pdf_export_path, jpegs_path, pages_to_ignore_set)
                progress_msg.info(f"PDF Conversion Complete. {len(jpeg_list)} pages converted.")

            total_pages = len(jpeg_list)

            with st.spinner("Running Model..."):
                for idx, (page_num, img_path) in enumerate(jpeg_list):
                    progress_msg.info(f"Processing Page {page_num} of {pdf_filename}")
                    image_cv = cv2.imread(img_path)
                    res = model(img_path)

                    page_results = []
                    callout_count = 0
                    annotated_frame = image_cv.copy()

                    for r in res:
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()

                        for i, box in enumerate(xyxy):
                            cval = round(float(confs[i]), 2)
                            x1, y1, x2, y2 = map(int, box)
                            pad = 5
                            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                            x2, y2 = min(image_cv.shape[1], x2 + pad), min(image_cv.shape[0], y2 + pad)

                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 25)
                            cropped = image_cv[y1:y2, x1:x2]
                            text = perform_ocr_with_easyocr(cropped)
                            wl = extract_wl_number(text)

                            # Put WL # above box
                            if wl:
                                font_scale = 6
                                font_thickness = 25
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                (text_width, text_height), _ = cv2.getTextSize(wl, font, font_scale, font_thickness)
                                text_org_x = x1
                                text_org_y = max(y1 - 10, text_height)
                                cv2.putText(
                                    annotated_frame, wl,
                                    (text_org_x, text_org_y),
                                    font,
                                    font_scale,
                                    (255, 0, 0),
                                    font_thickness
                                )
                            callout_count += 1
                            page_results.append([
                                wo_num, wl, text, page_num, cval, img_path
                            ])

                    st.session_state.all_annotated_images.append((pdf_filename, page_num, annotated_frame))

                    df_page = pd.DataFrame(
                        page_results,
                        columns=["WO#", "WL #", "Callout", "PDF Page", "Conf", "JPEG Path"]
                    )

                    unique_key = f"{pdf_filename}_page_{page_num}"
                    st.session_state.all_page_data[unique_key] = df_page

                    progress_msg.info(
                        f"Page {page_num}/{total_pages} of {pdf_filename} Processed: "
                        f"{callout_count} Callout(s) Found"
                    )

            progress_msg.success(f"Done processing {pdf_filename}!")
            end_time = time.time()
            st.write(f"**Processing Time for {pdf_filename}:** {round(end_time - start_time, 2)} seconds")

            # Create and save Excel for this PDF in its own folder
            pdf_data_frames = [df for k, df in st.session_state.all_page_data.items() if k.startswith(pdf_filename)]
            if pdf_data_frames:
                combined_pdf_df = pd.concat(pdf_data_frames, ignore_index=True)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                excel_path = os.path.join(pdf_run_dir, f"{wo_num}_{timestamp}.xlsx")
                combined_pdf_df.to_excel(excel_path, index=False)
                st.success(f"Excel saved: {excel_path}")

if st.session_state.all_annotated_images:
    st.subheader("All Annotated Images and Tables")
    st.session_state.all_annotated_images.sort(key=lambda x: (x[0], x[1]))
    for (pdf_name, page_num, image) in st.session_state.all_annotated_images:
        st.image(image, caption=f"{pdf_name} - Page {page_num}", use_container_width=True)

# Removed the old "Export All Data to Excel" block since Excel now lives near each PDF
