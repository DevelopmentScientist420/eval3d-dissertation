from openai import OpenAI
import os
from PIL import Image
import base64
from uuid import uuid4

client = OpenAI()

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

def gpt4(image, prompt):
    
    if os.path.exists(image):
        img = Image.open(image)
        b64 = encode_image_to_base64(img)
        img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail="low")
    else:
        img_struct = {
                "url": image,
            }
    
    
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
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
    ],
    max_tokens=2000,
    )
    
    return response.choices[0].message.content


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

def get_gpt4_resp(image_url, questions):
    questions_in_prompt = "\n".join([f"Q[{i}]: {q}" for i, q in questions.items()])
    
    prompt = vqa_prompt.replace("INSERT_QUESTIONS_HERE", questions_in_prompt)
    
    return gpt4(image_url, prompt)


def parse_gpt4_vqa_answers(gpt4_response):
    answers = {}
    for line in gpt4_response.split("\n"):
        if line.startswith("A["):
            q_num = int(line[2:line.find("]:")])
            answer = line[line.find(":")+2:]
            answers[q_num] = answer.strip()
    return answers

def get_vqa_answers(image, questions):
    resp = get_gpt4_resp(image, questions)
    return parse_gpt4_vqa_answers(resp)