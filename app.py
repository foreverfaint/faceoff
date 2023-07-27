from pathlib import Path
import hashlib
import time

import streamlit as st
import cv2
import insightface
import numpy as np
from streamlit_image_select import image_select


@st.cache_resource
def get_face_analyser():
    # st.info("Start to load face analyser model.")
    # t_1 = time.time()
    fa = insightface.app.FaceAnalysis(
        name="buffalo_l", providers=["CPUExecutionProvider"]
    )
    fa.prepare(ctx_id=0)
    # t_2 = time.time()
    # cost = round(t_2 - t_1, 2)
    # st.success(f"Finished load face analyser model (cost: {cost} secs)!")
    return fa


@st.cache_resource
def get_face_swapper():
    # st.info("Start to load face swapper model.")
    # t_1 = time.time()
    model_path = str(Path("/root/.insightface/models/inswapper_128.onnx"))
    fs = insightface.model_zoo.get_model(model_path, providers=["CPUExecutionProvider"])
    # t_2 = time.time()
    # cost = round(t_2 - t_1, 2)
    # st.success(f"Finished load face swapper model (cost: {cost} secs)!")
    return fs


def _rgb_to_bgr(rgb_img):
    return rgb_img[..., ::-1]


def _bgr_to_rgb(bgr_img):
    return bgr_img[..., ::-1]


def _get_hash(bytes) -> str:
    h = hashlib.md5()
    h.update(bytes)
    return h.hexdigest()


def _saves_uploaded_file(uploaded_file) -> Path:
    data = uploaded_file.read()
    file_name = Path("./.output") / (_get_hash(data) + ".jpg")
    with open(file_name, mode="wb") as f:
        f.write(data)
    return file_name


def _loads_bgr_image(file: Path):
    return cv2.imread(str(file))


def _choose_largest_face(faces):
    max_face = None
    max_area = -1
    for face in faces:
        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box
        area = (y2 - y1) * (x2 - x1)
        if area > max_area:
            max_area = area
            max_face = face
    return max_face


def get_faces(bgr_img):
    return get_face_analyser().get(bgr_img)


def render_faces_to_rgb_image(bgr_img, faces):
    if faces is None or len(faces) == 0:
        return _bgr_to_rgb(bgr_img)
    return _bgr_to_rgb(get_face_analyser().draw_on(bgr_img, faces))


def compose_rgb_img(bgr_res, faces_in_template, faces_in_portrait):
    fs = get_face_swapper()
    selected_face_in_portrait = _choose_largest_face(faces_in_portrait)

    for face in faces_in_template[:3]:
        bgr_res = fs.get(
            bgr_res,
            face,
            selected_face_in_portrait,
            paste_back=True,
        )

    return _bgr_to_rgb(bgr_res)


def get_selected_bgr_image(selected):
    if selected is None:
        return None

    if isinstance(selected, np.ndarray):
        rgb_img = selected
        return _rgb_to_bgr(rgb_img)

    if isinstance(selected, str):
        if selected.startswith("/app"):
            # /app/static/templates/3.jpg -> static/templates/3.jpg
            selected = "/".join(selected.split("/")[2:])
        return _loads_bgr_image(selected)

    return None


@st.cache_resource
def get_portrait_urls():
    return [
        f"/app/static/portraits/{x}"
        for x in [
            "10.jpg",
            "2.jpg",
            "3.jpg",
            "4.jpg",
            "5.jpg",
            "18.jpg",
            "11.jpg",
            "14.jpg",
            "15.jpg",
        ]
    ]


def get_portrait_urls_with_uploaded_file(uploaded_file):
    if uploaded_file:
        return [str(_saves_uploaded_file(uploaded_file))] + get_portrait_urls()
    return get_portrait_urls()


@st.cache_resource
def get_template_urls():
    return [
        f"/app/static/templates/{x}"
        for x in [
            "3.jpg",
            "4.jpg",
            "5.jpg",
            "6.jpg",
            "7.jpg",
            "8.jpg",
            "9.jpg",
            "10.jpg",
            "11.jpg",
            "12.png",
            "13.jpg",
            "14.jpg",
            "15.jpg",
            "16.jpg",
            "17.jpg",
            "18.jpg",
            "19.jpg",
            "20.jpg",
            "21.jpg",
        ]
    ]


if "stage" not in st.session_state:
    st.session_state.stage = 0


if "selected_portrait_img_url" not in st.session_state:
    st.session_state.selected_portrait_img_url = get_portrait_urls()[0]


if "selected_template_img_url" not in st.session_state:
    st.session_state.selected_template_img_url = get_template_urls()[0]


def set_state(i):
    st.session_state.stage = i


def render_stage_0():
    st.header("Choose Portrait")
    st.button("Next: Choose Template", on_click=set_state, args=[1], use_container_width=True, type='primary')

    uploaded_file = st.file_uploader(
        "Upload an image (.png, .jpg) to Portrait list",
        type=["png", "jpg"],
        accept_multiple_files=False,
        on_change=None,
        label_visibility="visible",
    )

    selected_portrait_img_url = image_select(
        "Click to choose an image as portrait",
        get_portrait_urls_with_uploaded_file(uploaded_file),
    )
    st.session_state.selected_portrait_img_url = selected_portrait_img_url


def render_stage_1():
    st.header("Choose Template")
    st.button("Next: Face Check", on_click=set_state, args=[2], use_container_width=True, type='primary')
    st.button("Prev: Choose Portrait", on_click=set_state, args=[0], use_container_width=True, type='secondary')

    selected_template_img_url = image_select(
        "Click to choose an image as template",
        get_template_urls(),
    )
    st.session_state.selected_template_img_url = selected_template_img_url


def detect_faces(session_key):
    img_url = st.session_state.get(session_key)
    if img_url is None:
        return None, [], -1

    bgr_img = get_selected_bgr_image(img_url)

    t_1 = time.time()
    faces = get_faces(bgr_img)
    t_2 = time.time()

    cost = round(t_2 - t_1, 2)
    return bgr_img, faces, cost


def render_stage_2():
    (
        selected_portrait_bgr_img,
        faces_in_portrait,
        cost_detect_faces_in_portrait,
    ) = detect_faces("selected_portrait_img_url")
    (
        selected_template_bgr_img,
        faces_in_template,
        cost_detect_faces_in_template,
    ) = detect_faces("selected_template_img_url")
    disabled = len(faces_in_portrait) == 0 or len(faces_in_template) == 0

    st.header("Face Check")
    st.button(
        "Next: Compose",
        on_click=set_state,
        args=[3],
        disabled=disabled,
        help="Can't detect a face to compose" if disabled else "", use_container_width=True, type='primary'
    )
    st.button("Prev: Choose Template", on_click=set_state, args=[1], use_container_width=True, type='secondary')

    col_2, col_3 = st.columns(2)
    with col_2:
        if selected_portrait_bgr_img is not None:
            st.header(
                f"Source (Face detection costs {cost_detect_faces_in_portrait} secs)"
            )
            st.image(
                render_faces_to_rgb_image(selected_portrait_bgr_img, faces_in_portrait)
            )

    with col_3:
        if selected_template_bgr_img is not None:
            st.header(
                f"Target (Face detection costs  {cost_detect_faces_in_template} secs)"
            )
            st.image(
                render_faces_to_rgb_image(selected_template_bgr_img, faces_in_template)
            )


def render_stage_3():
    st.header("Face Check")
    st.button("Reset", on_click=set_state, args=[0], use_container_width=True, type='primary')

    selected_portrait_img_url = st.session_state.get("selected_portrait_img_url")
    selected_portrait_bgr_img = get_selected_bgr_image(selected_portrait_img_url)
    faces_in_portrait = get_faces(selected_portrait_bgr_img)

    selected_template_img_url = st.session_state.get("selected_template_img_url")
    selected_template_bgr_img = get_selected_bgr_image(selected_template_img_url)
    faces_in_template = get_faces(selected_template_bgr_img)

    st.info("Start composing!")
    t_1 = time.time()
    composed_rgb_img = compose_rgb_img(
        selected_template_bgr_img,
        faces_in_template,
        faces_in_portrait,
    )
    t_2 = time.time()
    st.header(f"Composed Result (Face swap costs {round(t_2 - t_1, 2)} secs)")
    st.image(composed_rgb_img)
    st.success("Finished composing!")

    col_2, col_3 = st.columns(2)
    with col_2:
        st.image(_bgr_to_rgb(selected_portrait_bgr_img))

    with col_3:
        st.image(_bgr_to_rgb(selected_template_bgr_img))


if st.session_state.stage == 0:
    render_stage_0()
elif st.session_state.stage == 1:
    render_stage_1()
elif st.session_state.stage == 2:
    render_stage_2()
elif st.session_state.stage == 3:
    render_stage_3()
