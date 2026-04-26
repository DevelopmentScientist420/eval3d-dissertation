import os
from io import BytesIO
from PIL import Image
import base64
from uuid import uuid4
from urllib.request import urlopen
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

QWEN2_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

_model = None
_processor = None


def _get_model_and_processor():
    global _model, _processor
    if _model is None or _processor is None:
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN2_MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        _processor = AutoProcessor.from_pretrained(QWEN2_MODEL_NAME)
    return _model, _processor


def _load_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if not isinstance(image, str):
        raise TypeError("image must be a path, URL, data URI, or PIL Image")

    if os.path.exists(image):
        return Image.open(image).convert("RGB")

    if image.startswith("data:image") and "base64," in image:
        _, b64_data = image.split("base64,", 1)
        return Image.open(BytesIO(base64.b64decode(b64_data))).convert("RGB")

    if image.startswith("http://") or image.startswith("https://"):
        with urlopen(image, timeout=30) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")

    raise FileNotFoundError(f"Image source not found or unsupported: {image}")

# VQA with local image
def encode_image_to_base64(img, target_size=-1):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    tmp = os.path.join('/tmp', str(uuid4()) + '.jpg')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img.save(tmp)
    with open(tmp, 'rb') as image_file:
        image_data = image_file.read()
    ret = base64.b64encode(image_data).decode('utf-8')
    os.remove(tmp)
    return ret

def qwen2(image, prompt):

    if isinstance(image, str) and os.path.exists(image):
        img = Image.open(image)
        b64 = encode_image_to_base64(img)
        img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail="low")
    else:
        img_struct = {
                "url": image,
            }
    
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": img_struct,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    model, processor = _get_model_and_processor()

    user_content = messages[0]["content"]
    image_payload = user_content[0]["image_url"]
    image_source = image_payload["url"] if isinstance(image_payload, dict) else image_payload
    input_image = _load_image(image_source)

    qwen_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_content[1]["text"]},
            ],
        }
    ]

    text = processor.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = processor(
        text=[text],
        images=[input_image],
        return_tensors="pt",
    )
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0] if output_text else ""


vqa_prompt = \
"""Given a image and multiple questions, answer the questions with yes or no based on the image. Follow the formatting examples below.

Formatting example:
Questions:
Q[1]: Is there a cat?
Q[2]: Is the cat black?

Answers:
Q[1]: Is there a cat?
A[1]: Yes
Q[2]: Is the cat black?
A[2]: No

Now answer the following questions based on the image:
Questions:
INSERT_QUESTIONS_HERE

Answers:
"""

def get_qwen2_resp(image_url, questions):
    questions_in_prompt = "\n".join([f"Q[{i}]: {q}" for i, q in questions.items()])
    
    prompt = vqa_prompt.replace("INSERT_QUESTIONS_HERE", questions_in_prompt)
    
    return qwen2(image_url, prompt)


def parse_qwen2_vqa_answers(qwen2_response):
    answers = {}
    for line in qwen2_response.split("\n"):
        if line.startswith("A["):
            q_num = int(line[2:line.find("]:")])
            answer = line[line.find(":")+2:]
            answers[q_num] = answer.strip()
    return answers

def get_vqa_answers(image, questions):
    resp = get_qwen2_resp(image, questions)
    return parse_qwen2_vqa_answers(resp)
def qwen2_text(prompt):
    model, processor = _get_model_and_processor()

    qwen_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = processor(
        text=[text],
        return_tensors="pt",
    )
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0] if output_text else ""


import json

TIFA_PROMPT = """You are a precise AI that breaks down text prompts for 3D models into evaluation questions.
You must output a JSON object containing:
- "prompt": the original text prompt
- "questions": a dictionary of yes/no questions to verify the generated 3D model
- "scene_graph": a dictionary of entities, attributes, and actions
- "question_dependencies": a dictionary mapping question IDs to lists of prerequisite question IDs

Here is an example of the EXACT expected behavior.

Example Input:
Prompt: a dog chasing a ball

Example Output:
{
    "prompt": "a dog chasing a ball",
    "questions": {
        "1": "Is there a dog?",
        "2": "Is there a ball?",
        "3": "Is the dog chasing the ball?"
    },
    "scene_graph": {
        "1": "entity - dog",
        "2": "entity - ball",
        "3": "action - (dog, chase, ball)"
    },
    "question_dependencies": {
        "1": [0],
        "2": [0],
        "3": [1, 2]
    }
}

Now do the same for the following prompt. Do NOT use placeholders like "..." or "entity - ...". Provide the actual questions and scene graph based on the prompt. Return ONLY the JSON object.

Input:
Prompt: {user_prompt}

Output:
"""

def generate_tifa_questions(prompt):
    full_prompt = TIFA_PROMPT.replace("{user_prompt}", prompt)
    resp = qwen2_text(full_prompt)
    try:
        # cleanup in case the model added ```json wrappers
        resp = resp.strip()
        if resp.startswith("```json"):
            resp = resp[7:]
        if resp.startswith("```"):
            resp = resp[3:]
        if resp.endswith("```"):
            resp = resp[:-3]
        resp = resp.strip()
        return json.loads(resp)
    except Exception as e:
        print("Failed to parse JSON from Qwen2:", e)
        print("Raw response:", resp)
        raise e
