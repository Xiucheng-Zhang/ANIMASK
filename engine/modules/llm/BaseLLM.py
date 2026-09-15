# Modified file; the original work is under Apache-2.0, see README (License).
from abc import ABC, abstractmethod

# shared call context for the ANIMASK_CALL_LOG. The engine stamps
# the current round here; adapters copy it into each log line. Per-call kind
# lives on the llm INSTANCE (`_resim_kind`, set by the calling method) so
# interleaved role/world calls cannot mislabel each other.
CALL_CTX = {"round": None}


def set_call_ctx(**kw):
    CALL_CTX.update(kw)


class BaseLLM(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def initialize_message(self):
        pass

    @abstractmethod    
    def ai_message(self, payload):
        pass

    @abstractmethod
    def system_message(self, payload):
        pass

    @abstractmethod
    def user_message(self, payload):
        pass

    @abstractmethod
    def get_response(self):
        pass

    @abstractmethod
    def print_prompt(self):
        pass


