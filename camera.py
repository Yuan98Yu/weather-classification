import requests
import base64


def get_img(URL: str):
    # 访问服务
    response = requests.post(URL)
    # 获取服务器返回的图片，字节流返回
    result = response.content
    # 字节转换成图片
    img_file = base64.b64decode(result)

    return img_file
