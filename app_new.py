import streamlit as st



import requests
from faster_whisper import WhisperModel
import os
import cv2
import tempfile
import json
from datetime import datetime
from io import BytesIO
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.set_page_config(
    page_title="CrimeVision AI",
    page_icon="🔍",
    layout="wide"
)

# Session-backed evidence store. It lasts for this browser session.
if "evidence_history" not in st.session_state:
    st.session_state.evidence_history = []
if "current_evidence" not in st.session_state:
    st.session_state.current_evidence = []

st.title("🔍 CrimeVision AI")
st.subheader("Intelligent Multimodal Forensic Analysis Platform")

st.write(
    "Analyze image, video, and audio evidence "
    "with AI-assisted forensic tools."
)

st.info("Welcome! Your forensic analysis workspace is ready.")

st.markdown("### Select Evidence Type")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🖼️ Image Analysis")
    st.write("Upload and examine image evidence.")
    uploaded_image = st.file_uploader(

        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
     st.image(uploaded_image, caption="Uploaded Evidence")

     st.write("File Name:", uploaded_image.name)
     st.write("File Size:", round(uploaded_image.size / 1024, 2), "KB")

    if uploaded_image is not None:
     if st.button("🔍 Detect Objects"):
        with st.spinner("Analyzing image..."):
            image = Image.open(uploaded_image).convert("RGB")
            image_array = np.array(image)

            results = model(image_array)
            detected_image = results[0].plot()

            image_detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    label = model.names.get(class_id, str(class_id))
                    image_detections.append({
                        "label": label,
                        "confidence": confidence
                    })

            image_record = {
                "type": "Image",
                "filename": uploaded_image.name,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "findings": [
                    f'{item["label"]} ({item["confidence"]:.0%})'
                    for item in image_detections
                ],
                "details": image_detections
            }
            st.session_state.current_evidence = [
                item for item in st.session_state.current_evidence
                if not (item.get("type") == "Image" and item.get("filename") == uploaded_image.name)
            ]
            st.session_state.current_evidence.append(image_record)
            st.session_state.evidence_history.insert(0, image_record)

            st.image(
                detected_image,
                caption="Object Detection Results"
            )
            st.success("Object detection completed!")
    else:
             st.info("Please upload an image to begin analysis.")
        
with col2:
    st.subheader("🎬 Video Analysis")
    st.write("Detect objects in important video frames.")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("🔍 Analyze Important Frames", key="analyze_video"):
            temp_path = None
            cap = None

            try:
                suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as temp_file:
                    temp_file.write(uploaded_video.getbuffer())
                    temp_path = temp_file.name

                cap = cv2.VideoCapture(temp_path)

                if not cap.isOpened():
                    st.error("Could not open this video.")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    st.write(f"**Total frames:** {total_frames}")

                    if total_frames <= 0:
                        st.error("No readable frames found.")
                    else:
                        # Sample at most 12 frames across the video
                        sample_count = min(12, total_frames)

                        frame_indices = np.linspace(
                            0,
                            total_frames - 1,
                            num=sample_count,
                            dtype=int
                        )

                        important_frames = []
                        progress = st.progress(0)

                        for i, frame_index in enumerate(frame_indices):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                            success, frame = cap.read()

                            if not success:
                                progress.progress((i + 1) / sample_count)
                                continue

                            # OpenCV BGR -> RGB
                            frame_rgb = cv2.cvtColor(
                                frame, cv2.COLOR_BGR2RGB
                            )

                            results = model(frame_rgb, verbose=False)
                            result = results[0]

                            if result.boxes is not None and len(result.boxes) > 0:
                                detected_frame = result.plot()
                                detected_frame = cv2.cvtColor(
                                    detected_frame, cv2.COLOR_BGR2RGB
                                )

                                timestamp = (
                                    frame_index / fps if fps and fps > 0 else 0
                                )

                                detections = []

                                for box in result.boxes:
                                    class_id = int(box.cls[0].item())
                                    confidence = float(box.conf[0].item())
                                    label = model.names.get(
                                        class_id, str(class_id)
                                    )

                                    detections.append(
                                        f"{label} ({confidence:.0%})"
                                    )

                                important_frames.append({
                                    "image": detected_frame,
                                    "timestamp": timestamp,
                                    "detections": detections
                                })

                            progress.progress((i + 1) / sample_count)

                        cap.release()
                        cap = None

                        st.caption(
                            f"Sampled {sample_count} frames out of "
                            f"{total_frames} total frames. "
                            "This is a sample, not a frame-by-frame analysis."
                        )

                        video_findings = []
                        for item in important_frames:
                            video_findings.extend(item["detections"])

                        video_record = {
                            "type": "Video",
                            "filename": uploaded_video.name,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "findings": video_findings,
                            "details": [
                                {
                                    "timestamp": f'{int(item["timestamp"] // 60):02d}:{item["timestamp"] % 60:05.2f}',
                                    "detections": item["detections"]
                                }
                                for item in important_frames
                            ]
                        }
                        st.session_state.current_evidence = [
                            item for item in st.session_state.current_evidence
                            if not (item.get("type") == "Video" and item.get("filename") == uploaded_video.name)
                        ]
                        st.session_state.current_evidence.append(video_record)
                        st.session_state.evidence_history.insert(0, video_record)

                        if important_frames:
                            st.success(
                                f"Found objects in "
                                f"{len(important_frames)} sampled frames."
                            )

                            for number, item in enumerate(
                                important_frames, start=1
                            ):
                                minutes = int(item["timestamp"] // 60)
                                seconds = item["timestamp"] % 60

                                st.markdown(
                                    f"### Important Frame {number} — "
                                    f"{minutes:02d}:{seconds:05.2f}"
                                )

                                st.image(
                                    item["image"],
                                    use_container_width=True
                                )

                                st.write(
                                    "**Detected objects:** "
                                    + ", ".join(item["detections"])
                                )
                        else:
                            st.warning(
                                "No objects were detected in the sampled "
                                "frames. Try another video."
                            )

            except Exception as e:
                st.error(f"Video analysis failed: {e}")

            finally:
                if cap is not None:
                    cap.release()

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    else:
        st.info("Upload a video to begin analysis.")


with col3:
    st.subheader("🎙️ Audio Analysis")
    st.write("Upload audio to transcribe spoken words.")

    uploaded_audio = st.file_uploader(
        "Upload an audio recording",
        type=["wav", "mp3", "m4a", "ogg", "flac", "aac"],
        key="audio_upload"
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        if st.button("🎧 Transcribe Audio", key="transcribe_audio"):

            audio_temp_path = None

            try:
                suffix = (
                    os.path.splitext(uploaded_audio.name)[1] or ".wav"
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as audio_file:
                    audio_file.write(uploaded_audio.getbuffer())
                    audio_temp_path = audio_file.name

                with st.spinner(
                    "AI is listening to the recording... "
                    "The first run may take longer."
                ):
                    @st.cache_resource
                    def load_audio_model():
                        return WhisperModel(
                            "base",
                            device="cpu",
                            compute_type="int8"
                        )

                    audio_model = load_audio_model()
                    segments, info = audio_model.transcribe(
                         audio_temp_path,
                         beam_size=5,
                          vad_filter=True
)
                    transcript_segments = list(segments)

                st.success("Audio transcription completed!")

                st.write(f"**Detected language:** {info.language}")
                st.write(
                    f"**Language confidence:** "
                    f"{info.language_probability:.0%}"
                )

                transcript_text = " ".join(
                    segment.text.strip()
                    for segment in transcript_segments
                )

                audio_record = {
                    "type": "Audio",
                    "filename": uploaded_audio.name,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "findings": [transcript_text] if transcript_text.strip() else ["No clear speech detected"],
                    "details": [
                        {
                            "start": f"{int(segment.start) // 60:02d}:{int(segment.start) % 60:02d}",
                            "end": f"{int(segment.end) // 60:02d}:{int(segment.end) % 60:02d}",
                            "text": segment.text.strip()
                        }
                        for segment in transcript_segments
                    ],
                    "language": info.language,
                    "language_confidence": float(info.language_probability)
                }
                st.session_state.current_evidence = [
                    item for item in st.session_state.current_evidence
                    if not (item.get("type") == "Audio" and item.get("filename") == uploaded_audio.name)
                ]
                st.session_state.current_evidence.append(audio_record)
                st.session_state.evidence_history.insert(0, audio_record)

                if transcript_text.strip():
                    st.markdown("### 📝 Audio Transcript")
                    st.text_area(
                        "Recognized speech",
                        value=transcript_text,
                        height=250
                    )

                    st.download_button(
                        label="⬇️ Download Transcript",
                        data=transcript_text,
                        file_name="audio_transcript.txt",
                        mime="text/plain"
                    )

                    with st.expander("View timestamps"):
                        for segment in transcript_segments:
                            start = int(segment.start)
                            end = int(segment.end)

                            st.write(
                                f"**{start // 60:02d}:"
                                f"{start % 60:02d} – "
                                f"{end // 60:02d}:"
                                f"{end % 60:02d}** "
                                f"{segment.text.strip()}"
                            )
                else:
                    st.warning(
                        "No clear speech was detected in this recording."
                    )

            except Exception as e:
                st.error(f"Audio analysis failed: {e}")

            finally:
                if (
                    audio_temp_path
                    and os.path.exists(audio_temp_path)
                ):
                    os.remove(audio_temp_path)

    else:
        st.info("Upload an audio recording to begin analysis.")



# ==================== CONSOLIDATED FORENSIC WORKSPACE ====================
st.divider()
st.header("🧾 Forensic Workspace")
st.caption(
    "Reports summarize the outputs of the selected AI tools. "
    "Object detection and transcription are aids to review, not proof of a crime."
)

records = st.session_state.current_evidence

report_col, timeline_col = st.columns([1, 1])
with report_col:
    st.subheader("📋 Detailed Forensic Report")
    if not records:
        st.info("Analyze at least one image, video, or audio file to build a report.")
    else:
        st.write(f"**Evidence items analyzed:** {len(records)}")
        for record in records:
            with st.expander(f'{record["type"]}: {record["filename"]}'):
                st.write(f'**Analysis time:** {record["time"]}')
                if record.get("language"):
                    st.write(
                        f'**Detected language:** {record["language"]} '
                        f'({record.get("language_confidence", 0):.0%} confidence)'
                    )
                if record["type"] == "Video":
                    if record["details"]:
                        for frame in record["details"]:
                            st.write(f'**Timestamp {frame["timestamp"]}:** ' + ", ".join(frame["detections"]))
                    else:
                        st.write("No objects detected in sampled frames.")
                elif record["type"] == "Audio":
                    st.text_area(
                        "Transcript / finding",
                        value="\n".join(record.get("findings", [])),
                        height=120,
                        key=f'report_audio_{record["filename"]}'
                    )
                    for segment in record.get("details", []):
                        st.caption(f'{segment["start"]}–{segment["end"]}: {segment["text"]}')
                else:
                    st.write("**Detected objects:**")
                    if record.get("findings"):
                        for finding in record["findings"]:
                            st.write(f"• {finding}")
                    else:
                        st.write("No objects detected.")

with timeline_col:
    st.subheader("🕒 Evidence Timeline")
    if not records:
        st.info("Your findings will appear here in chronological order.")
    else:
        timeline_items = []
        for record in records:
            if record["type"] == "Video":
                for detail in record.get("details", []):
                    timeline_items.append({
                        "sort": detail.get("timestamp", ""),
                        "evidence": record["filename"],
                        "kind": "Video frame",
                        "finding": ", ".join(detail.get("detections", []))
                    })
            elif record["type"] == "Audio":
                for detail in record.get("details", []):
                    timeline_items.append({
                        "sort": detail.get("start", ""),
                        "evidence": record["filename"],
                        "kind": "Audio segment",
                        "finding": detail.get("text", "")
                    })
            else:
                timeline_items.append({
                    "sort": record.get("time", ""),
                    "evidence": record["filename"],
                    "kind": "Image finding",
                    "finding": ", ".join(record.get("findings", [])) or "No objects detected"
                })
        for item in sorted(timeline_items, key=lambda x: x["sort"]):
            st.markdown(
                f'**{item["sort"]}** · `{item["kind"]}` · **{item["evidence"]}**'
            )
            st.write(item["finding"] or "No text/object finding recorded.")

st.divider()
assistant_col, export_col = st.columns([1, 1])

with assistant_col:
    st.subheader("🤖 AI Forensic Assistant")
    st.caption("Ask about the results currently recorded in this session.")
    question = st.text_input(
        "Your question",
        placeholder="e.g., Which objects were detected in the video?",
        key="forensic_question"
    )
    if st.button("Answer", key="forensic_answer"):
            if not records:
                st.info("Analyze evidence first, then ask a question.")
            elif not question.strip():
                st.warning("Type a question first.")
            else:
                all_findings = []
            for record in records:
                if record.get("findings"):
                    all_findings.append(
                        f'{record["type"]} "{record["filename"]}": '
                        + "; ".join(record["findings"])
                    )
                if record["type"] == "Video":
                    for detail in record.get("details", []):
                        all_findings.append(
                            f'Video timestamp {detail.get("timestamp")}: '
                            + ", ".join(detail.get("detections", []))
                        )

            evidence_summary = "\n".join(all_findings) or "No findings recorded."

            try:
                api_key = st.secrets["GROQ_API_KEY"]
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are CrimeVision AI, a forensic assistant. "
                                    "Answer using the supplied evidence. Never invent "
                                    "findings or declare anyone guilty. Explain that "
                                    "results must be verified against original evidence."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Evidence findings:\n{evidence_summary}\n"
                                    f"Question: {question}"
                                ),
                            },
                        ],
                        "temperature": 0.2,
                    },
                    timeout=30,
                )

                if response.ok:
                    answer = response.json()["choices"][0]["message"]["content"]
                    st.info(answer)
                else:
                    st.error(f"Groq API error: {response.status_code}")

            except KeyError:
                st.error("GROQ_API_KEY is missing from Streamlit Secrets.")
            except requests.RequestException:
                st.error("Could not connect to Groq. Please try again.")
            except (ValueError, IndexError):
                st.error("Could not read Groq's response. Please try again.")
            except Exception as e:
                st.error(f"AI Assistant error: {e}")


with export_col:
    st.subheader("📄 Export Forensic Report")
    st.write("Create a PDF summary of the evidence analyzed in this session.")
    if not records:
        st.info("Analyze at least one evidence item to enable PDF export.")
    else:
        if st.button("Build PDF Report", key="build_pdf"):
            buffer = BytesIO()
            styles = getSampleStyleSheet()
            story = [
                Paragraph("CrimeVision AI — Forensic Analysis Report", styles["Title"]),
                Spacer(1, 12),
                Paragraph(
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    styles["Normal"]
                ),
                Spacer(1, 12),
                Paragraph(
                    "Scope note: This report summarizes automated object detection and/or speech transcription. "
                    "It is not a determination of identity, guilt, or legal fact.",
                    styles["BodyText"]
                ),
                Spacer(1, 12)
            ]
            table_data = [["Evidence type", "File name", "Analysis time", "Findings"]]
            for record in records:
                findings = record.get("findings", [])
                if record["type"] == "Video":
                    findings = [
                        f'{detail.get("timestamp", "")}: {", ".join(detail.get("detections", []))}'
                        for detail in record.get("details", [])
                    ]
                elif record["type"] == "Audio":
                    findings = record.get("findings", [])
                summary = "; ".join(findings) or "No findings detected"
                table_data.append([
                    record.get("type", ""),
                    record.get("filename", ""),
                    record.get("time", ""),
                    summary[:500]
                ])
            table = Table(table_data, repeatRows=1, colWidths=[75, 105, 95, 210])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243447")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
            ]))
            story.append(table)
            doc = SimpleDocTemplate(
                buffer, pagesize=letter,
                rightMargin=30, leftMargin=30, topMargin=36, bottomMargin=36
            )
            doc.build(story)
            st.download_button(
                "⬇️ Download Forensic PDF",
                data=buffer.getvalue(),
                file_name="CrimeVision_Forensic_Report.pdf",
                mime="application/pdf",
                key="download_forensic_pdf"
            )

st.divider()
st.subheader("🗂️ Evidence History")
st.caption(
    "History is stored in Streamlit session state and resets when the session is restarted. "
    "Do not treat it as permanent evidence storage."
)
if st.session_state.evidence_history:
    for idx, record in enumerate(st.session_state.evidence_history[:20], start=1):
        with st.expander(f'{idx}. {record["type"]} — {record["filename"]} — {record["time"]}'):
            st.write("Findings:", "; ".join(record.get("findings", [])) or "None")
else:
    st.info("No evidence history yet. Analyze an image, video, or audio file to add a record.")

