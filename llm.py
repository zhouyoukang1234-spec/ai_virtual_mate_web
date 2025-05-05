# 在llm.py最顶部添加
import requests  # 🔥 关键修复
from vlm import *

try:
    from letta import create_client, LLMConfig, EmbeddingConfig
    from letta.schemas.memory import ChatMemory
except:
    pass
with open('data/db/memory.db', 'r', encoding='utf-8') as memory_file:
    try:
        openai_history = json.load(memory_file)
    except:
        openai_history = []
spark_history = []
sf_url = "https://api.siliconflow.cn/v1"


def chat_preprocess(msg):
    # 统一加载配置文件
    with open('data/set/more_set.json', 'r', encoding='utf-8') as f:
        more_config = json.load(f)
    # 语音指令关键词读取
    VOICE_KEYWORDS = more_config.get("HAAI关键词", [])
    for kw in VOICE_KEYWORDS:
        if kw in msg:
            command = msg.replace(kw, "", 1).strip()
            return send_ha_voice_command(command)
            
    # 文本指令关键词读取
    for kw in more_config.get("HA文本指令关键词", []):
        if kw in msg:
            command = msg.replace(kw, "", 1).strip()
            result = send_ha_command(command)
            return f"{result}"
    
    # 原有图像识别逻辑不变
    try:
        content = "图像识别已关闭"
        if ("屏幕" in msg or "画面" in msg or "图像" in msg or "看到" in msg or "看见" in msg or "照片" in msg or "摄像头" in msg or "图片" in msg) and img_menu.get() != "关闭图像识别":
            if "图片" in msg:
                if os.path.exists("data/cache/cache.png"):
                    if img_menu.get() == "GLM-4V-Flash":
                        content = glm_4v_photo(msg)
                    elif img_menu.get() == "本地Ollama VLM":
                        content = ollama_vlm_photo(msg)
                    elif img_menu.get() == "本地QwenVL整合包":
                        content = qwen_vlm_photo(msg)
                    elif img_menu.get() == "本地GLM-V整合包":
                        content = glm_v_photo(msg)
                    elif img_menu.get() == "本地Janus整合包":
                        content = janus_photo(msg)
                    elif img_menu.get() == "自定义API-VLM":
                        content = custom_vlm_photo(msg)
                    notice(f"{mate_name}识别了上传的图片")
                    os.remove("data/cache/cache.png")
                else:
                    content = "请先点击右下方按钮上传图片"
                    notice("请先点击右下方按钮上传图片")
            elif "屏幕" in msg or "画面" in msg or "图像" in msg:
                if img_menu.get() == "GLM-4V-Flash":
                    content = glm_4v_screen(msg)
                elif img_menu.get() == "本地Ollama VLM":
                    content = ollama_vlm_screen(msg)
                elif img_menu.get() == "本地QwenVL整合包":
                    content = qwen_vlm_screen(msg)
                elif img_menu.get() == "本地GLM-V整合包":
                    content = glm_v_screen(msg)
                elif img_menu.get() == "本地Janus整合包":
                    content = janus_screen(msg)
                elif img_menu.get() == "自定义API-VLM":
                    content = custom_vlm_screen(msg)
                notice(f"{mate_name}捕获了屏幕，调用[电脑屏幕识别]")
            elif "看到" in msg or "看见" in msg or "照片" in msg or "摄像头" in msg:
                if img_menu.get() == "GLM-4V-Flash":
                    content = glm_4v_cam(msg)
                elif img_menu.get() == "本地Ollama VLM":
                    content = ollama_vlm_cam(msg)
                elif img_menu.get() == "本地QwenVL整合包":
                    content = qwen_vlm_cam(msg)
                elif img_menu.get() == "本地GLM-V整合包":
                    content = glm_v_cam(msg)
                elif img_menu.get() == "本地Janus整合包":
                    content = janus_cam(msg)
                elif img_menu.get() == "自定义API-VLM":
                    content = custom_vlm_cam(msg)
                notice(f"{mate_name}拍了照片，调用[摄像头识别]")
        else:
            content = chat_llm(prompt, msg)
            notice(f"收到{mate_name}回复")
        with open('data/db/memory.db', 'w', encoding='utf-8') as memory_file:
            json.dump(openai_history, memory_file, ensure_ascii=False, indent=4)
        return content
    except Exception as e:
        notice(f"发生错误，错误详情：{e}")
        return f"发生错误，错误详情：{e}"


def chat_llm(tishici, msg):  # 大语言模型聊天
    try:
        if llm_menu.get() == "讯飞星火Lite":
            spark_client = OpenAI(base_url="https://spark-api-open.xf-yun.com/v1", api_key=spark_key)
            spark_history.append({"role": "user", "content": f"{tishici}。我的问题是：{msg}"})
            messages = []
            messages.extend(spark_history)
            completion = spark_client.chat.completions.create(model="general", messages=messages)
            spark_history.append({"role": "assistant", "content": completion.choices[0].message.content})
            return completion.choices[0].message.content
        elif llm_menu.get() == "GLM-4-Flash":
            glm_client = OpenAI(base_url=glm_url, api_key=glm_key)
            openai_history.append({"role": "user", "content": msg})
            messages = [{"role": "system", "content": tishici}]
            messages.extend(openai_history)
            completion = glm_client.chat.completions.create(model="glm-4-flash-250414", messages=messages)
            openai_history.append({"role": "assistant", "content": completion.choices[0].message.content})
            return completion.choices[0].message.content
        elif llm_menu.get() == "GLM-Z1-Flash":
            glm_client = OpenAI(base_url=glm_url, api_key=glm_key)
            openai_history.append({"role": "user", "content": msg})
            messages = [{"role": "system", "content": tishici}]
            messages.extend(openai_history)
            completion = glm_client.chat.completions.create(model="glm-z1-flash", messages=messages)
            openai_history.append({"role": "assistant", "content": completion.choices[0].message.content})
            return completion.choices[0].message.content
        elif llm_menu.get() == "DeepSeek-R1-7B":
            ds_client = OpenAI(base_url=sf_url, api_key=sf_key)
            openai_history.append({"role": "user", "content": msg})
            messages = [{"role": "system", "content": tishici}]
            messages.extend(openai_history)
            completion = ds_client.chat.completions.create(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                                                           messages=messages)
            openai_history.append({"role": "assistant", "content": completion.choices[0].message.content})
            return completion.choices[0].message.content
        elif llm_menu.get() == "通义千问2.5-7B":
            qwen_client = OpenAI(base_url=sf_url, api_key=sf_key)
            openai_history.append({"role": "user", "content": msg})
            messages = [{"role": "system", "content": tishici}]
            messages.extend(openai_history)
            completion = qwen_client.chat.completions.create(model="Qwen/Qwen2.5-7B-Instruct",
                                                             messages=messages)
            openai_history.append({"role": "assistant", "content": completion.choices[0].message.content})
            return completion.choices[0].message.content
        elif llm_menu.get() == "InternLM2.5-7B":
            internlm_client = OpenAI(base_url=sf_url, api_key=sf_key)
            openai_history.append({"role": "user", "content": msg})
            messages = [{"role": "system", "content": tishici}]
            messages.extend(openai_history)
            completion = internlm_client.chat.completions.create(model="internlm/internlm2_5-7b-chat",
                                                                 messages=messages)
            openai_history.append({"role": "assistant", "content": completion.choices[0].message.content})
            return completion.choices[0].message.content
        elif llm_menu.get() == "腾讯混元Lite":
            hunyuan_client = OpenAI(base_url="https://api.hunyuan.cloud.tencent.com/v1", api_key=hy_key)
            openai_history.append({"role": "user", "content": msg})
            messages = [{"role": "system", "content": tishici}]
            messages.extend(openai_history)
            completion = hunyuan_client.chat.completions.create(model="hunyuan-lite", messages=messages)
            openai_history.append({"role": "assistant", "content": completion.choices[0].message.content})
            return completion.choices[0].message.content
        elif llm_menu.get() == "本地Qwen整合包":
            api = f"http://{local_server_ip}:8088/llm/?p={tishici}&q={msg}"
            try:
                res = rq.get(api).json()["answer"]
                return res
            except Exception as e:
                return f"本地Qwen整合包API服务器未开启，错误详情：{str(e)[0:100]}"
        elif llm_menu.get() == "本地LM Studio":
            try:
                lmstudio_client = OpenAI(base_url=f"http://{local_server_ip}:{lmstudio_port}/v1", api_key="lm-studio")
                openai_history.append({"role": "user", "content": msg})
                messages = [{"role": "system", "content": tishici}]
                messages.extend(openai_history)
                completion = lmstudio_client.chat.completions.create(model="", messages=messages)
                openai_history.append({"role": "assistant", "content": completion.choices[0].message.content})
                return completion.choices[0].message.content
            except Exception as e:
                return f"本地LM Studio软件API服务未开启，错误详情：{e}"
        elif llm_menu.get() == "本地Ollama LLM":
            try:
                try:
                    rq.get(f'http://{local_server_ip}:{ollama_port}')
                except:
                    Popen(f"ollama pull {ollama_model_name}", shell=False)
                ollama_client = Client(host=f'http://{local_server_ip}:{ollama_port}')
                openai_history.append({"role": "user", "content": msg})
                messages = [{"role": "system", "content": tishici}]
                messages.extend(openai_history)
                response = ollama_client.chat(model=ollama_model_name, messages=messages)
                openai_history.append({"role": "assistant", "content": response['message']['content']})
                return response['message']['content']
            except Exception as e:
                return f"本地Ollama LLM配置错误，错误详情：{e}"
        elif llm_menu.get() == "本地RWKV运行器":
            try:
                rwkv_client = OpenAI(base_url=f"http://{local_server_ip}:8000/v1", api_key="rwkv")
                openai_history.append({"role": "user", "content": msg})
                messages = [{"role": "system", "content": tishici}]
                messages.extend(openai_history)
                completion = rwkv_client.chat.completions.create(model="rwkv", messages=messages)
                openai_history.append({"role": "assistant", "content": completion.choices[0].message.content})
                return completion.choices[0].message.content
            except Exception as e:
                return f"本地RWKV Runner软件API服务未开启，错误详情：{e}"
        elif llm_menu.get() == "本地OpenVINO":
            api = f"http://{local_server_ip}:8087/openvino/?p={tishici}&q={msg}"
            try:
                res = rq.get(api).json()["answer"]
                return res
            except Exception as e:
                return f"本地OpenVINO整合包API服务器未开启，错误详情：{str(e)[0:100]}"
        elif llm_menu.get() == "Dify聊天助手":
            try:
                res = chat_dify(msg)
                return res
            except Exception as e:
                return f"本地Dify聊天助手配置错误，错误详情：{e}"
        elif llm_menu.get() == "AnythingLLM":
            try:
                res = chat_anything_llm(msg)
                return res
            except Exception as e:
                return f"本地AnythingLLM知识库配置错误，错误详情：{e}"
        elif llm_menu.get() == "Letta长期记忆":
            res = chat_letta(msg)
            return res
        else:
            try:
                custom_client = OpenAI(base_url=custom_url, api_key=custom_key)
                openai_history.append({"role": "user", "content": msg})
                messages = [{"role": "system", "content": tishici}]
                messages.extend(openai_history)
                completion = custom_client.chat.completions.create(model=custom_model, messages=messages)
                openai_history.append({"role": "assistant", "content": completion.choices[0].message.content})
                return completion.choices[0].message.content
            except Exception as e:
                return f"自定义API配置错误，错误详情：{e}"
    except Exception as e:
        notice(f"{llm_menu.get()}不可用，请前往软件设置正确配置云端AI Key，错误详情：{e}")
        return f"{llm_menu.get()}不可用，请前往软件设置正确配置云端AI Key，错误详情：{e}"


def chat_dify(msg):  # Dify聊天助手
    headers = {"Authorization": f"Bearer {dify_key}", "Content-Type": "application/json"}
    data = {"query": msg, "inputs": {}, "response_mode": "blocking", "user": username, "conversation_id": None}
    res = rq.post(f"http://{dify_ip}/v1/chat-messages", headers=headers, data=json.dumps(data))
    res = res.json()['answer'].strip()
    return res


def chat_anything_llm(msg):  # AnythingLLM知识库
    url = f"http://{local_server_ip}:3001/api/v1/workspace/{anything_llm_ws}/chat"
    headers = {"Authorization": f"Bearer {anything_llm_key}", "Content-Type": "application/json"}
    data = {"message": msg}
    res = rq.post(url, json=data, headers=headers)
    return res.json().get("textResponse")


def chat_letta(msg):  # Letta长期记忆
    answer = "Letta长期记忆服务拥挤，请一段时间后再试"
    try:
        client = create_client()
        with open('data/db/letta.db', 'r', encoding='utf-8') as f:
            agent_state_id = f.read()
        if len(agent_state_id) < 10:
            agent_state = client.create_agent(
                memory=ChatMemory(persona=prompt, human=f"Name: {username}"),
                llm_config=LLMConfig.default_config(model_name="letta"),
                embedding_config=EmbeddingConfig.default_config(model_name="text-embedding-ada-002"))
            with open('data/db/letta.db', 'w', encoding='utf-8') as f:
                f.write(agent_state.id)
            agent_state_id = agent_state.id
        response = client.send_message(
            agent_id=agent_state_id, role="user", message=f"[{current_time()}]{msg}")
        result = response.messages
        for message in result:
            if message.message_type == 'tool_call_message':
                function_arguments = message.tool_call.arguments
                if function_arguments:
                    arguments_dict = json.loads(function_arguments)
                    answer = arguments_dict.get("message")
                    break
    except:
        answer = "Letta长期记忆出现兼容性问题暂不可用，可更换其他对话模型"
    return answer


def clear_chat():  # 清除对话记录
    global openai_history, spark_history
    if messagebox.askokcancel(f"清除{mate_name}的记忆和聊天记录",
                              f"您确定要清除{mate_name}的记忆和聊天记录吗？\n如有需要可先点击🔼导出记录再开启新对话"):
        output_box.delete("1.0", "end")
        openai_history, spark_history = [], []
        with open('data/db/letta.db', 'w', encoding="utf-8") as f:
            f.write("0")
        with open('data/db/memory.db', 'w', encoding='utf-8') as f:
            f.write("")
        notice("记忆和聊天记录已清空")

# 新增在llm.py中
def send_ha_command(command):
    try:
        with open('data/set/more_set.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        ha_config = {
            "url": config.get("HomeAssistant服务器IP", ""),
            "token": config.get("HomeAssistant Token", ""),
            "text_entity": config.get("文本指令实体ID", "")
        }
        
        response = requests.post(
            f"{ha_config['url']}/api/services/text/set_value",
            headers={"Authorization": f"Bearer {ha_config['token']}"},
            json={"entity_id": ha_config["text_entity"], "value": command},
            timeout=5
        )

        result_list = response.json()
        if isinstance(result_list, list) and len(result_list) > 0:
            result = f"执行成功：{result_list[0].get('state', '操作完成')}".replace("{lv=stt}", command)
        else:
            result = "指令已执行"
            
        notice(f"收到{mate_name}回复")  # 新增通知 <mcsymbol name="notice" filename="llm.py" path="f:\虚拟ai伙伴\web版ai伙伴开发\枫云AI虚拟伙伴Web版v3.0\llm.py" startline="1" type="function"></mcsymbol>
        return result
        
    except Exception as e:
        return f"操作异常：{str(e)}"  # 错误信息也去除前缀

def send_ha_voice_command(text):
    try:
        with open('data/set/more_set.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        ha_config = {
            "url": config.get("HomeAssistant服务器IP", ""),
            "token": config.get("HomeAssistant Token", ""),
            "voice_agent": config.get("语音API实体ID", "")
        }
        
        response = requests.post(
            f"{ha_config['url']}/api/conversation/process",
            headers={"Authorization": f"Bearer {ha_config['token']}"},
            json={"agent_id": ha_config["voice_agent"], "text": text, "language": "zh-CN"},
            timeout=20
        )
        
        notice(f"收到{mate_name}回复")  # 新增通知
        return response.json()['response']['speech']['plain']['speech']

    except Exception as e:
        return f"语音指令失败: {str(e)}"



def clean_chat_web():  # 清除对话记录
    global openai_history, spark_history
    openai_history, spark_history = [], []
