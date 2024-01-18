from graph_of_thoughts import controller, operations
from graph_of_thoughts import prompter, parser

# 定义Prompter
class TranslationPrompter(prompt.Prompter):
    def __init__(self, language_graph, source_language, target_language):
        self.language_graph = language_graph
        self.source_language = source_language
        self.target_language = target_language

    def prompt(self, input_data):
        # 根据语种图和其他信息生成提示
        similarity_score = self.language_graph.graph.get(self.source_language, {}).get(self.target_language, 0)
        prompt = f"Given that the similarity between {self.source_language} and {self.target_language} is {similarity_score}, "
        prompt += f"and the initial translation of '{input_data['source_sentence']}' is '{input_data['initial_translation']}', "
        prompt += "please provide a better translation."
        return prompt

    def generate_prompt(
        self, num_branches: int, original: str, current: str, method: str, **kwargs
    ) -> str:
        # original: 源句子
        # current: 初步翻译
        # kwargs["true_translation"]: 真实翻译
        # kwargs["prior_relation"]: 语言之间的先验关系
        prompt = f"""
        <Instruction> Given the source sentence '{original}', its preliminary translation '{current}', the true translation '{kwargs["true_translation"]}', and the prior relation between the source language and the target language '{kwargs["prior_relation"]}', please generate a better translation. </Instruction>
        Input: {original}
        Preliminary Translation: {current}
        """
        return prompt

# 定义Parser
class TranslationParser(parser.Parser):
    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        new_states = []
        for text in texts:
            # 假设GPT-3.5返回的翻译结果是在"Output:"后面
            start_index = text.find("Output:") + len("Output:")
            end_index = text.find("\n", start_index)
            if end_index == -1:
                end_index = len(text)
            translation = text[start_index:end_index].strip()
            new_state = state.copy()
            new_state["current"] = translation
            new_states.append(new_state)
        return new_states
        
    def parse(self, response):
        # 解析GPT-3.5的输出以获取翻译结果
        return response['choices'][0]['text'].strip()


# 定义语种图结构
class LanguageGraph:
    def __init__(self):
        self.graph = {}  # {language: {connected_language: similarity_score}}

    def add_language(self, language):
        if language not in self.graph:
            self.graph[language] = {}

    def add_similarity(self, lang1, lang2, similarity):
        self.add_language(lang1)
        self.add_language(lang2)
        self.graph[lang1][lang2] = similarity
        self.graph[lang2][lang1] = similarity

# 定义parser
class TranslationParser:
    def parse(self, response):
        # 解析GPT-3.5的输出以获取翻译结果
        return response['choices'][0]['text'].strip()


def main():
    # 创建语种图
    lang_graph = LanguageGraph()
    # 添加语种和它们之间的相似度（示例）
    lang_graph.add_similarity('lang1', 'lang2', 0.8)

    # 使用GoT框架构建GoO
    gop = operations.GraphOfOperations()
    gop.append_operation(operations.Generate())  # 生成提示
    gop.append_operation(operations.Score())  # 评估提示的质量

    # 输入：源句子、初步翻译、真实翻译
    input_data = {
        "source_sentence": "your_source_sentence",
        "initial_translation": "your_initial_translation",
        "true_translation": "your_true_translation"
    }

    # 问题参数
    problem_parameters = {
        "language_graph": lang_graph,
        "source_language": "your_source_language",
        "target_language": "English"
    }
    prompter = TranslationPrompter(lang_graph, 'lang1', 'English')
    prompt = prompter.prompt(input_data)

    parser = TranslationParser()
    # 假设response是GPT-3.5的输出
    # translation = parser.parse(response)

    # 使用GoT框架与GPT-3.5交互
    lm = controller.ChatGPT("config.json", model_name="chatgpt")
    parser = TranslationParser()
    ctrl = controller.Controller(lm, gop, input_data, parser, problem_parameters)
    ctrl.run()


if __name__ == '__main__':
    main()