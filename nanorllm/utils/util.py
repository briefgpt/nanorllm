from typing import Any


def render_prompt_messages(prompt_messages: list[dict[str, Any]]) -> str:
    rendered_messages: list[str] = []

    for message in prompt_messages or []:
        role = str(message.get("role", "user")).strip().upper()
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        rendered_messages.append(f"<{role}>\n{content.strip()}")

    prompt_text = "\n\n".join(part for part in rendered_messages if part).strip()
    return prompt_text



def render_prompt_for_completion(messages: list[dict[str, Any]]) -> dict[str, str]:
    '''
    1. 历史messages 处理(role 之间用\n\n， role和content之间用\n)
    2. 衔接最新回复（<Assistant>\n)
    '''
    
    prompt_text= render_prompt_messages(messages).strip()

    if prompt_text:
        prompt_text = f"{prompt_text}\n\n<Assistant>\n"
    else:
        prompt_text = "<Assistant>\n"
    return prompt_text