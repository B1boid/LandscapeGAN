import json
import os
import cherrypy
import yaml
import torch

import psutil
process = psutil.Process(os.getpid())  # for monitoring and debugging purposes

config = yaml.safe_load(open("config.yml"))

def get_device():
    cuda = torch.cuda.is_available()
    print(cuda)
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device

device = get_device()

from main import LandscapeGan
landscapeGan = LandscapeGan()



def process_api_request(body):
    """
    This methos is for extracting json parameters and processing the actual api request.
    All api format and logic is to be kept in here.
    :param body:
    :return:
    """

    # mode = image/tags
    mode = body.get('mode')

    # image = base64 encoded picture
    image = body.get('image')

    # tags = "tag1 tag2 ..."
    tags = body.get('tags')

    # need_tt = True/False
    need_tt = body.get('need_tt')

    warnings = []
    mode_allowed = ["image", "tags"]
    if mode not in mode_allowed:
        mode = mode_allowed[0]
        warnings.append("Mode is not in: " + str(mode_allowed) + ". Mode was set to: " + mode + ".")

    try:
        result = landscapeGan.generate(mode, image=image, tags=tags, need_tt=need_tt)
    except Exception as e:
        result = {'error': str(e)}
    if len(warnings) != 0:
        result['warnings'] = warnings

    return json.dumps(result)


class ApiServerController(object):
    @cherrypy.expose('/health')
    def health(self):
        result = {
            "status": "OK",
            "info": {
                "mem": "{0:.3f} MiB".format(process.memory_info().rss / 1024.0 / 1024.0),
                "cpu": process.cpu_percent(),
                "threads": len(process.threads())
            }
        }
        return json.dumps(result).encode("utf-8")

    @cherrypy.expose('/generate-landscape')
    def generate_landscape(self):
        cl = cherrypy.request.headers['Content-Length']
        raw = cherrypy.request.body.read(int(cl))
        body = json.loads(raw)
        return process_api_request(body).encode("utf-8")


if __name__ == '__main__':
    cherrypy.tree.mount(ApiServerController(), '/')
    cherrypy.config.update({
        'server.socket_port': config["app"]["port"],
        'server.socket_host': config["app"]["host"],
        'server.thread_pool': config["app"]["thread_pool"],
        'log.access_file': "access1.log",
        'log.error_file': "error1.log",
        'log.screen': True,
        'tools.response_headers.on': True,
        'tools.encode.encoding': 'utf-8',
        'tools.response_headers.headers': [('Content-Type', 'application/json;encoding=utf-8')],
    })

    try:
        cherrypy.engine.start()
        print("Start")
        cherrypy.engine.block()
    except KeyboardInterrupt:
        cherrypy.engine.stop()
