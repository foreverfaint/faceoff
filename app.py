from pathlib import Path
import hashlib
import time

import streamlit as st
import cv2
import insightface
import numpy as np
from streamlit_image_select import image_select
from gfpgan.utils import GFPGANer


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
    model_path = "./models/inswapper_128.onnx"
    fs = insightface.model_zoo.get_model(model_path, providers=["CPUExecutionProvider"])
    # t_2 = time.time()
    # cost = round(t_2 - t_1, 2)
    # st.success(f"Finished load face swapper model (cost: {cost} secs)!")
    return fs


@st.cache_resource
def get_face_enhancer():
    model_path = "./models/GFPGANv1.4.pth"
    # todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
    return GFPGANer(
        model_path=model_path,
        arch="clean",
        channel_multiplier=2,
        upscale=1,
        device="cpu",
    )


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
    st.session_state.stage = -1


if "selected_portrait_img_url" not in st.session_state:
    st.session_state.selected_portrait_img_url = get_portrait_urls()[0]


if "selected_template_img_url" not in st.session_state:
    st.session_state.selected_template_img_url = get_template_urls()[0]


def set_state(i):
    st.session_state.stage = i


def render_stage_auth():
    def _check():
        passcode = st.session_state.get("passcode")
        if passcode != "1983":
            st.error("Wrong passcode")
        else:
            set_state(0)

    st.text_input("Enter the passcode", on_change=_check, key="passcode")


def render_stage_0():
    st.header("Choose Portrait")
    st.button(
        "Next: Choose Template",
        on_click=set_state,
        args=[1],
        use_container_width=True,
        type="primary",
    )

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
    st.button(
        "Next: Configure",
        on_click=set_state,
        args=[2],
        use_container_width=True,
        type="primary",
    )
    st.button(
        "Prev: Choose Portrait",
        on_click=set_state,
        args=[0],
        use_container_width=True,
        type="secondary",
    )

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


if "multi_faces" not in st.session_state:
    st.session_state.multi_faces = "largest"


if "enhance_mode" not in st.session_state:
    st.session_state.enhance_mode = "disable_enhance"


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

    st.header("Configure")
    st.button(
        "Next: Compose",
        on_click=set_state,
        args=[3],
        disabled=disabled,
        help="Can't detect a face to compose" if disabled else "",
        use_container_width=True,
        type="primary",
    )
    st.button(
        "Prev: Choose Template",
        on_click=set_state,
        args=[1],
        use_container_width=True,
        type="secondary",
    )

    st.radio(
        "Enhancement Mode",
        ("disable_enhance", "enhance_image", "enhance_face"),
        format_func=lambda x: "Disable Enhance (fast, low quality)"
        if x == "disable_enhance"
        else (
            "Enhance the whole image (very slow, high quality)"
            if x == "enhance_image"
            else "Enhance face by face"
        ),
        index=1,
        key="enhance_mode",
    )
    st.radio(
        "How to handle multiple faces in the template",
        ("largest", "top3", "all"),
        format_func=lambda x: "The largest face (fast)"
        if x == "largest"
        else ("The first 3 faces" if x == "top3" else "All the faces (slow)"),
        index=0,
        key="multi_faces",
    )

    col_2, col_3 = st.columns(2)
    with col_2:
        if selected_portrait_bgr_img is not None:
            st.subheader(
                f"Source (Face detection costs {cost_detect_faces_in_portrait} secs)"
            )
            st.image(
                render_faces_to_rgb_image(selected_portrait_bgr_img, faces_in_portrait)
            )

    with col_3:
        if selected_template_bgr_img is not None:
            st.subheader(
                f"Target (Face detection costs  {cost_detect_faces_in_template} secs)"
            )
            st.image(
                render_faces_to_rgb_image(selected_template_bgr_img, faces_in_template)
            )


def compose_rgb_img(
    bgr_res,
    faces_in_template,
    faces_in_portrait,
    enhance_mode,
    multi_faces,
):
    fe = get_face_enhancer()
    selected_face_in_portrait = _choose_largest_face(faces_in_portrait)

    swapped_faces = faces_in_template
    if multi_faces == "top3":
        swapped_faces = faces_in_template[:3]
    elif multi_faces == "largest":
        swapped_faces = [_choose_largest_face(faces_in_template)]

    fs = get_face_swapper()
    for face in swapped_faces:
        bgr_res = fs.get(
            bgr_res,
            face,
            selected_face_in_portrait,
            paste_back=True,
        )

    # face enhance isn't stable, it may raise
    #   File "/root/.cache/pypoetry/virtualenvs/faceoff-L2WRRFYm-py3.10/lib/python3.10/site-packages/gfpgan/utils.py", line 145, in enhance
    #    restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
    #  File "/root/.cache/pypoetry/virtualenvs/faceoff-L2WRRFYm-py3.10/lib/python3.10/site-packages/facexlib/utils/face_restoration_helper.py", line 291, in paste_faces_to_input_image
    #    assert len(self.restored_faces) == len(
    # AssertionError: length of restored_faces and affine_matrices are different.
    #  Stopping...
    #
    try:
        if enhance_mode == "enhance_face":
            fe = get_face_enhancer()
            for face in swapped_faces:
                box = face.bbox.astype(int)
                x1, y1, x2, y2 = box
                _, _, enhanced_face = fe.enhance(bgr_res[y1:y2, x1:x2], paste_back=True)
                if enhanced_face is not None:
                    bgr_res[y1:y2, x1:x2] = enhanced_face
        elif enhance_mode == "enhance_image":
            fe = get_face_enhancer()
            _, _, bgr_res = fe.enhance(bgr_res, paste_back=True)
    except Exception as e:
        import traceback

        traceback.print_tb(e.__traceback__)

    return _bgr_to_rgb(bgr_res)


def compose_faces():
    enhance_mode = st.session_state.get("enhance_mode")
    multi_faces = st.session_state.get("multi_faces")

    selected_portrait_img_url = st.session_state.get("selected_portrait_img_url")
    selected_portrait_bgr_img = get_selected_bgr_image(selected_portrait_img_url)
    faces_in_portrait = get_faces(selected_portrait_bgr_img)

    selected_template_img_url = st.session_state.get("selected_template_img_url")
    selected_template_bgr_img = get_selected_bgr_image(selected_template_img_url)
    faces_in_template = get_faces(selected_template_bgr_img)

    print(
        f"portrait_img_url={selected_portrait_img_url}, template_img_url={selected_template_img_url}, multi_faces={multi_faces}, enhance_mode={enhance_mode}"
    )

    placeholder = st.empty()
    with placeholder.container():
        t_1 = time.time()
        composed_rgb_img = compose_rgb_img(
            selected_template_bgr_img,
            faces_in_template,
            faces_in_portrait,
            enhance_mode,
            multi_faces,
        )
        t_2 = time.time()
        st.subheader(f"Composed Result (Face swap costs {round(t_2 - t_1, 2)} secs)")
        st.image(composed_rgb_img)

    col_2, col_3 = st.columns(2)
    with col_2:
        st.subheader("Portrait")
        st.image(_bgr_to_rgb(selected_portrait_bgr_img))

    with col_3:
        st.subheader("Template")
        st.image(_bgr_to_rgb(selected_template_bgr_img))


def render_stage_3():
    st.header("Compose")
    st.button(
        "Prev: Configure",
        on_click=set_state,
        args=[2],
        use_container_width=True,
        type="primary",
    )

    compose_faces()


if st.session_state.stage == -1:
    render_stage_auth()
elif st.session_state.stage == 0:
    render_stage_0()
elif st.session_state.stage == 1:
    render_stage_1()
elif st.session_state.stage == 2:
    render_stage_2()
elif st.session_state.stage == 3:
    render_stage_3()
