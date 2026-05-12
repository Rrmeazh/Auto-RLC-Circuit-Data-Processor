import os
from openai import OpenAI

class ModelChat:
    def __init__(
            self,
            api_key: str | None = None,
            base_url: str | None = None,
            model: str | None = None
        ) -> None:
        '''
        初始化ModelChat对象，设置API密钥、基础URL和模型名称
        '''
        self.model = model
        base_url = base_url
        client_options = {"api_key": api_key}
        if base_url:
            client_options["base_url"] = base_url
        self.client = OpenAI(**client_options)

    def response(
            self,
            system_prompt: str,
            report_prompt: str,
            result: str
        ) -> str:
        '''
        根据实验数据处理结果生成实验报告分析

        Args:
            system_prompt: 系统提示，描述模型的角色和任务
            report_prompt: 用户提示，说明需要生成什么样的报告
            result: 实验数据处理结果文本
        '''
        prompt = f"""{report_prompt}{result}""".strip()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


if __name__ == "__main__":
    script_path = os.path.dirname(os.path.abspath(__file__))
    data2_path = os.path.join(script_path, "data", "data2.csv")
    with open(os.path.join(script_path, "result", "result_2.md"), "r", encoding="utf-8") as f:
        result = f.read()

    with open(os.path.join(script_path, "modelConfig.json"), "r", encoding="utf-8") as f:
        import json
        config = json.load(f)[0]
    chat = ModelChat(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"]
    )
    report = chat.response(
        system_prompt=config["system_prompt"],
        report_prompt=config["report_prompt"],
        result=result
    )
    print(report)
