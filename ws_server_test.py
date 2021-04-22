# WS server example
import asyncio
import websockets
import json
# import base64
from io import BytesIO
from PIL import Image

from server_config import ws_web_server_config

count = 0


async def hello(websocket, path):
    data = await websocket.recv()
    data = json.loads(data)
    print(type(data))
    print(data)

    # greeting = f"Hello {name}!"

    # await websocket.send(greeting)
    # print(f"> {greeting}")


async def rec_img(websocket, path):
    global count
    data = await websocket.recv()
    # data = base64.b64decode(data)
    print('rec data!'.center(100, '='))
    print(type(data))
    stream = BytesIO(data)
    # img = Image.frombytes('RGB', (320, 320), data)
    img = Image.open(stream).convert('RGB')
    print(type(img))
    img.save(f'./test_{count}.png')
    count += 1


start_server = websockets.serve(hello, ws_web_server_config['ip'],
                                ws_web_server_config['port'])
# start_server = websockets.serve(rec_img, '0.0.0.0', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
