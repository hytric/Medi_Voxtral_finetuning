"""
다양한 음성-언어 태스크를 위한 템플릿 정의
"""

AUDIO_PLACEHOLDER = "<|audio|>"
TRANSCRIPTION_PLACEHOLDER = "<|transcription|>"

# 기본 템플릿들
TEMPLATES = {
    "transcription": {
        "user_template": (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "<|transcribe|>\n"
            f"{AUDIO_PLACEHOLDER}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "assistant_template": f"{TRANSCRIPTION_PLACEHOLDER}{{output}}<|eot_id|>",
        "system_prompt": "당신은 정확한 음성 전사 전문가입니다. 음성을 듣고 정확한 텍스트로 변환해주세요."
    },
    
    "qa": {
        "user_template": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n다음 질문에 답해주세요:\n\n{AUDIO_PLACEHOLDER}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_template": "{{output}}<|eot_id|>",
        "system_prompt": "당신은 도움이 되는 AI 어시스턴트입니다. 질문에 정확하고 유용한 답변을 제공해주세요."
    },
    
    "audio_info_qa": {
        "user_template": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n다음 음성 정보를 바탕으로 질문에 답해주세요:\n\n음성: {AUDIO_PLACEHOLDER}\n\n질문: {{text}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_template": "{{output}}<|eot_id|>",
        "system_prompt": "당신은 음성 정보를 분석하여 질문에 답변하는 전문가입니다. 음성의 내용을 정확히 파악하고 질문에 적절히 답변해주세요."
    },
    
    "medical_qa": {
        "user_template": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n다음 의료 관련 질문에 답해주세요:\n\n{AUDIO_PLACEHOLDER}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_template": "{{output}}<|eot_id|>",
        "system_prompt": "당신은 의료 전문가입니다. 의료 관련 질문에 정확하고 전문적인 답변을 제공해주세요."
    },
    
    "text_only_qa": {
        "user_template": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{{text}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_template": "{{output}}<|eot_id|>",
        "system_prompt": "당신은 도움이 되는 AI 어시스턴트입니다. 질문에 정확하고 유용한 답변을 제공해주세요."
    },
    
    "text_only_chat": {
        "user_template": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{{text}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_template": "{{output}}<|eot_id|>",
        "system_prompt": "당신은 친근하고 도움이 되는 AI 어시스턴트입니다. 자연스럽고 유용한 대화를 제공해주세요."
    },
    
    "custom": {
        "user_template": "{{user_template}}",
        "assistant_template": "{{assistant_template}}",
        "system_prompt": "{{system_prompt}}"
    }
}

def get_template(task_name=None, **kwargs):
    """
    태스크에 맞는 템플릿을 반환합니다.
    task_name이 None이면 모든 템플릿을 반환합니다.
    
    Args:
        task_name (str, optional): 태스크 이름. None이면 모든 템플릿 반환
        **kwargs: 템플릿에 전달할 추가 인자들
    
    Returns:
        dict: 템플릿 정보
    """
    if task_name is None:
        # 모든 템플릿 반환 (멀티태스크 학습용)
        return TEMPLATES
    
    if task_name not in TEMPLATES:
        raise ValueError(f"지원하지 않는 태스크: {task_name}. 지원하는 태스크: {list(TEMPLATES.keys())}")
    
    template = TEMPLATES[task_name].copy()
    
    # 추가 인자들로 템플릿 업데이트
    for key, value in kwargs.items():
        if key in template:
            template[key] = template[key].format(**kwargs)
    
    return template

def format_prompt(template, audio_placeholder=AUDIO_PLACEHOLDER, **kwargs):
    """
    템플릿을 실제 프롬프트로 포맷합니다.
    
    Args:
        template (dict): 템플릿 정보
        audio_placeholder (str): 오디오 플레이스홀더
        **kwargs: 포맷할 변수들
    
    Returns:
        dict: 포맷된 프롬프트
    """
    formatted = {}
    
    for key, value in template.items():
        if isinstance(value, str):
            # 오디오 플레이스홀더와 추가 변수들로 포맷
            format_kwargs = {"audio_placeholder": audio_placeholder, **kwargs}
            formatted[key] = value.format(**format_kwargs)
        else:
            formatted[key] = value
    
    return formatted
