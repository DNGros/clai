from typing import Optional


# pylint: disable=too-few-public-methods
class DockerMessage:
    def __init__(self, docker_command: str, message: str = ''):
        self.docker_command = docker_command
        self.message = message

# pylint: disable=too-few-public-methods
class DockerReply:
    def __init__(self, docker_reply: str, message: str = '', info: Optional[str] = None):
        self.docker_reply = docker_reply
        self.message = message
        self.info = info
