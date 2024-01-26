import requests
from .persistent import APIKEY, BASEURL, CLIENT_AGENT, MODEL_LIST
from invokeai.backend.util.logging import info, debug, warning, error

from typing import Literal, TypedDict, Union, Optional
from invokeai.app.invocations.baseinvocation import (
    BaseInvocationOutput,
    BaseInvocation,
    Input,
    InputField,
    UIComponent,
    InvocationContext,
    OutputField,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.primitives import StringOutput, ImageField
from invokeai.app.services.image_records.image_records_common import ImageCategory, ImageRecordChanges, ResourceOrigin
from pydantic import BaseModel, Field
import json
import random
import time
from PIL import Image

DEFAULT_REQUEST = {
  "prompt": "string",
  "params": {
    "sampler_name": "k_euler_a",
    "cfg_scale": 7.5,
    "denoising_strength": 0.75,
    "seed": "The little seed that could",
    "height": 512,
    "width": 512,
    "seed_variation": 1,
    "post_processing": [],
    "karras": False,
    "tiling": False,
    "hires_fix": False,
    "clip_skip": 1,
    "control_type": "canny",
    "image_is_control": False,
    "return_control_map": False,
    "facefixer_strength": 0.75,
    "loras": [
      {
        "name": "Magnagothica",
        "model": 1,
        "clip": 1,
        "inject_trigger": "string",
        "is_version": False
      }
    ],
    "tis": [
      {
        "name": "7808",
        "inject_ti": "prompt",
        "strength": 1
      }
    ],
    "special": {
      "*": {
        "additionalProp1": {},
        "additionalProp2": {},
        "additionalProp3": {}
      }
    },
    "steps": 30,
    "n": 1
  },
  "nsfw": False,
  "trusted_workers": False,
  "slow_workers": True,
  "censor_nsfw": False,
  "workers": [
    "string"
  ],
  "worker_blacklist": False,
  "models": [
    "string"
  ],
  "source_image": "string",
  "source_processing": "img2img",
  "source_mask": "string",
  "r2": True,
  "shared": False,
  "replacement_filter": True,
  "dry_run": False,
  "proxied_account": "string",
  "disable_batching": False
}


HEADERS = {"apikey": APIKEY, "Client-Agent": CLIENT_AGENT}

def image_request(request_options: dict) -> dict:
    """Creates a new image request on the horde.

    Args:
        custom_options (list[dict], optional): A list of dictionaries containing custom options to be added to the request. Defaults to [].

    Returns:
        dict: The response from the horde.
        {
            "id": "string",
            "kudos": 0,
            "message": "string"
        }
    """
    #options = DEFAULT_REQUEST
    
    url = BASEURL + 'generate/async'
    try:
        r: requests.Response = requests.post(url, headers=HEADERS, json=request_options)
    except:
        raise Exception('Failed during image request.')
    #good respose is code 202
    responsedict = r.json()
    if r.status_code != 202:
        error(f"Bad response from Horde: {r.status_code}")
        warning(f"Bad response from Horde: {responsedict['message']}")
        if r.status_code == 400:
            extra_errors = responsedict['errors']
            for error_info in extra_errors:
                warning(error_info)
        raise Exception(f"Error: Could not create image request. Status code {r.status_code}")
    return responsedict

SAMPLERS = Literal["k_euler_a", "k_dpmpp_sde", "k_dpmpp_2s_a", "DDIM", "dpmsolver", "k_euler", "lcm", "k_heun", "k_dpm_2", "k_lms", "k_dpm_fast", "k_dpm_adaptive", "k_dpm_2_a", "k_dpmpp_2m" ]

# example: "modelname (20)"
MODEL_LIST_TITLES = Literal[tuple([model['name'] for model in MODEL_LIST])]

#example: { "modelname (20)": "modelname" }
#MODEL_LIST_DICT = { f"{model['name']} ({model['count']})": model['name'] for model in MODEL_LIST }


class HordeGeneralSettings(BaseModel):
    settings: dict = Field(description="General Settings")
@invocation_output("horde_general_settings_output")
class HordeGeneralSettingsOutput(BaseInvocationOutput):
    """The output of the HordeGeneralSettingsInvocation."""
    general_settings_output: HordeGeneralSettings | None = OutputField(
        title="General Settings",
        description="General Settings",
    )

@invocation(
    "horde_general_settings",
    title="HORDE: General Settings",
    category="horde",
    tags=["horde", "settings"],
    version="1.0.0"
)
class HordeGeneralSettingsInvocation(BaseInvocation):
    """Gather advanced settings for AI Horde Image Requests."""
    sampler: SAMPLERS = InputField(
        description="The sampler to use when generating the image.",
        title="Sampler",
        input=Input.Direct,
        ui_order=0,
    )
    cfg_scale: float = InputField(
        description="The cfg_scale to use when generating the image.",
        title="CFG Scale",
        default=7.5,
        ui_order=1,
    )
    seed: int = InputField(
        description="The seed to use when generating the image.",
        title="Seed",
        default=0,
        ui_order=2,
    )
    seed_variation: int = InputField(
        description="Increment used if multiple images are requested.",
        title="Seed Variation",
        default=1,
        ui_order=3,
    )

    def invoke(self, context: InvocationContext) -> HordeGeneralSettingsOutput:
        """Invoke the advanced settings node."""
        return HordeGeneralSettingsOutput(
            general_settings_output = HordeGeneralSettings(
                settings={
                    "sampler_name": self.sampler,
                    "cfg_scale": self.cfg_scale,
                    "seed": self.seed,
                    "seed_variation": self.seed_variation,
                }
            )
        )

class HordeAdvancedSettings(BaseModel):
    settings: dict = Field(description="Advanced Settings")
    params: dict = Field(description="Advanced Params")
@invocation_output("horde_advanced_settings_output")
class HordeAdvancedSettingsOutput(BaseInvocationOutput):
    """The output of the HordeAdvancedSettingsInvocation."""
    advanced_settings_output: HordeAdvancedSettings | None = OutputField(
        title="Advanced Settings",
        description="Advanced Settings",
    )

@invocation(
    "horde_advanced_settings",
    title="HORDE: Advanced Settings",
    category="horde",
    tags=["horde", "settings"],
    version="1.0.0"
)
class HordeAdvancedSettingsInvocation(BaseInvocation):
    karras: bool = InputField(
        description="Set to True to enable karras noise scheduling tweaks.",
        title="Karras",
        default=False,
        ui_order=0,
    )
    tiling: bool = InputField(
        description="Set to True to create images that stitch together seamlessly.",
        title="Tiling",
        default=False,
        ui_order=1,
    )
    clip_skip: int = InputField(
        description="nth from last (1 is last) frame to use for clip skip.",
        title="Clip Skip",
        default=1,
        ge=1,
        le=12,
        ui_order=2,
    )
    nsfw: bool = InputField(
        description="Set to True to allow NSFW images.",
        title="NSFW",
        default=True,
        ui_order=3,
    )
    censor_nsfw: bool = InputField(
        description="Set to True to censor NSFW images.",
        title="Censor NSFW",
        default=False,
        ui_order=4,
    )
    trusted_workers: bool = InputField(
        description="When true, only trusted workers will serve this request.",
        title="Trusted Workers Only",
        default=False,
        ui_order=5,
    )
    slow_workers: bool = InputField(
        description="Allow slow workers. Costs extra kudos to disable",
        title="Allow Slow Workers",
        default=True,
        ui_order=6,
    )
    r2: bool = InputField(
        description="If True, the image will be sent via cloudflare r2 download link.",
        title="R2",
        default=True,
        ui_hidden=True, #only supports r2 image downloads for now
        ui_order=7,
    )
    shared: bool = InputField(
        description="If True, The image will be shared with LAION for improving their dataset.",
        title="Shared with LAION",
        default=False,
        ui_order=8,
    )
    replacement_filter: bool = InputField(
        description="If enabled, suspicious prompts are sanitized through a string replacement filter.",
        title="Replacement Filter",
        default=True,
        ui_order=9,
    )
    disable_batching: bool = InputField(
        description="If enabled, the request will not be batched.",
        title="Disable Batching",
        default=False,
        ui_order=10,
    )

    def invoke(self, context: InvocationContext) -> HordeAdvancedSettingsOutput:
        """Invoke the advanced settings node."""
        return HordeAdvancedSettingsOutput(
            advanced_settings_output = HordeAdvancedSettings(
                settings = {
                    "nsfw": self.nsfw,
                    "trusted_workers": self.trusted_workers,
                    "slow_workers": self.slow_workers,
                    "censor_nsfw": self.censor_nsfw,
                    "r2": self.r2,
                    "shared": self.shared,
                    "replacement_filter": self.replacement_filter,
                    "disable_batching": self.disable_batching,
                },
                params = {
                    "karras": self.karras,
                    "tiling": self.tiling,
                    "clip_skip": self.clip_skip,
                }
            )
        )



@invocation_output("horde_images_output")
class HordeImagesOutput(BaseInvocationOutput):
    """The output of the HordeImagesInvocation."""
    single_image_output: ImageField | None = OutputField(
        title="Single Image",
        description="First image returned from the horde.",
    )
    width: int = OutputField(
        title="Width",
        description="Width of the image.",
    )
    height: int = OutputField(
        title="Height",
        description="Height of the image.",
    )
    images_output: list[ImageField] | None = OutputField(
        title="All Images",
        description="If n>1, a list of all images returned from the horde. Use the Iterate node to loop through them.",
    )

"""
Design structure:
Request Image node will have:
    Direct Inputs:
        Positive prompt
        Negative prompt
        model
        sampler
        height
        width
        hires_fix
"""
@invocation(
    "horde_request_image",
    title="HORDE: Request Image",
    category="horde",
    tags=["horde", "Image", "generate"],
    version="1.0.0"
)
class HordeRequestImageInvocation(BaseInvocation):
    """Request an image from the horde."""
    positive_prompt: str = InputField(
        default="A picture of a cat.",
        description="A prompt that will be used to generate the image.",
        title="Positive Prompt",
        ui_component=UIComponent.Textarea,
        ui_order=0,
    )
    negative_prompt: str = InputField(
        default="",
        description="Content to avoid in the generated image.",
        title="Negative Prompt",
        ui_component=UIComponent.Textarea,
        ui_order=1,
    )
    model: MODEL_LIST_TITLES = InputField(
        description="The model to use when generating the image. Sorted by most workers.",
        title="Model",
        input=Input.Direct,
        ui_order=2,
    )
    height: int = InputField(
        description="The height of the image to generate.",
        title="Height",
        default=512,
        ge=64,
        le=3072,
        multiple_of=64,
        ui_order=3,
    )
    width: int = InputField(
        description="The width of the image to generate.",
        title="Width",
        default=512,
        ge=64,
        le=3072,
        multiple_of=64,
        ui_order=4,
    )
    hires_fix: bool = InputField(
        description="Whether to use the hires fix when generating the image.",
        title="Hires Fix",
        default=False,
        ui_order=5,
    )
    n: int = InputField(
        description="The number of images to generate.",
        title="Number of Images",
        default=1,
        ge=1,
        le=10,
        ui_order=6,
    )
    gen_settings: Optional[HordeGeneralSettings] = InputField(
        description="General Settings",
        default=None,
        title="General Settings",
        input=Input.Connection,
        ui_order=7,
    )
    adv_settings: Optional[HordeAdvancedSettings] = InputField(
        description="Advanced Settings",
        default=None,
        title="Advanced Settings",
        input=Input.Connection,
        ui_order=8,
    )

    timeout: int = InputField(
        description="Give up after waiting this many seconds.",
        title="Timeout",
        default=600,
        ge=1,
        le=600,
        ui_order=20,
    )


    def invoke(self, context: InvocationContext) -> HordeImagesOutput:
        """Invoke the request image node."""
        info("Requesting image from horde.")
        req = {} #instantiating the request dict
        req['prompt'] = f"{self.positive_prompt} ### {self.negative_prompt}"
        req['models'] = [self.model]
        req['params'] = {}
        req['params']['height'] = self.height
        req['params']['width'] = self.width
        req['params']['hires_fix'] = self.hires_fix
        req['params']['n'] = self.n

        #generate random integer string for seed if not provided
        req['params']['seed'] = str(random.randint(0, 10000000000))

        if self.gen_settings is not None:
            req['params'].update(self.gen_settings.settings)
            req['params']['seed'] = str(self.gen_settings.settings['seed']) #force to string
        if self.adv_settings is not None:
            req['params'].update(self.adv_settings.params)
            req.update(self.adv_settings.settings)
        
        req_UUID = image_request(req)['id']

        check_url = BASEURL + 'generate/check/' + req_UUID
        status_url = BASEURL + 'generate/status/' + req_UUID
        while True:
            response = requests.get(check_url, headers=HEADERS).json()
            info(response)
            if response["finished"] == req["params"]["n"]:
                break
            time.sleep(5)
        
        response = requests.get(status_url, headers = HEADERS).json()
        generations = response["generations"]
        img_list = []
        pil_list = []
        for generation in generations:
            # download the image as a PIL Image
            img = Image.open(requests.get(generation["img"], stream=True).raw)
            pil_list.append(img)
            # save the image
            image_dto = context.services.images.create(
                image=img,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                node_id=self.id,
                is_intermediate=True,
                session_id=context.graph_execution_state_id,
                workflow=context.workflow,
            )
            img_list.append(image_dto)

            imageField_list = []
            for image in img_list:
                imageField_list.append(ImageField(image_name=image.image_name))


        print(response)
        return HordeImagesOutput(
            single_image_output = ImageField(image_name=img_list[0].image_name),
            width = pil_list[0].width,
            height = pil_list[0].height,
            images_output = imageField_list,
        )