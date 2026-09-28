# -*- coding: utf-8 -*-
"""CalSem-HARQ: Calibrated Action-Conditional Semantic-Failure Risk for HARQ
in Generative Semantic Communication.

研究目标（FINAL_PROPOSAL）:
  生成式语义通信的 HARQ 决策应由「生成可靠性」而非「信道质量」驱动——
  用一个校准后的动作条件化语义失败风险头预测每个候选重传动作后的失败概率，
  再用效用最大化规则在失败风险下降与码率/时延间权衡选择动作（接受/语义细化/高保真细化）。
"""

__version__ = "0.1.0"
