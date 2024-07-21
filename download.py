import os

from paddleseg.utils.download import download_file_and_uncompress


model_urls = {
    'portrait_pp_humansegv1_lite_398x224_inference_model_with_softmax':
    'https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv1_lite_398x224_inference_model_with_softmax.zip',
    'portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax':
    'https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_smaller/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.zip',
    'human_pp_humansegv1_lite_192x192_inference_model_with_softmax':
    'https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_lite_192x192_inference_model_with_softmax.zip',
    'human_pp_humansegv2_lite_192x192_inference_model_with_softmax':
    'https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_lite_192x192_inference_model_with_softmax.zip',
    'human_pp_humansegv1_mobile_192x192_inference_model_with_softmax':
    'https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv1_mobile_192x192_inference_model_with_softmax.zip',
    'human_pp_humansegv2_mobile_192x192_inference_model_with_softmax':
    'https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax.zip',
}


if __name__ == '__main__':
    data_path = os.path.abspath(r'./inference_models')
    for model_name, url in model_urls.items():
        download_file_and_uncompress(
            url=url,
            savepath=data_path,
            extrapath=data_path,
            extraname=model_name)
    print('Download inference models finished.')
