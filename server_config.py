from pathlib import Path


ws_listen_host = '0.0.0.0'
ws_listen_port = 5678

img_save_dir = Path('/home/nwpu-2/wc_data/imgs/')
default_classes = ['cloudy', 'haze', 'rainy', 'snow', 'sunny', 'thunder']
ws_web_server_config = {'ip': 'localhost', 'port': 54789}

ws_web_server_url = f'ws://{ws_web_server_config["ip"]}:{ws_web_server_config["port"]}'
