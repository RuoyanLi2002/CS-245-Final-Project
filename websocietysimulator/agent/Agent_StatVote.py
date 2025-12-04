from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
import json
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
from websocietysimulator.llm import OpenAILLM
import logging
import re
import ast
import numpy as np

logging.basicConfig(level=logging.INFO)


class PlanningSmart(PlanningBase):
    """智能规划：结合多种策略，根据上下文动态生成最优计划"""

    def __init__(self, llm, interaction_tool=None, memory=None):
        """
        Initialize smart planning module

        Args:
            llm: LLM instance
            interaction_tool: InteractionTool for context analysis (optional)
            memory: Memory module for retrieving past successful plans (optional)
        """
        super().__init__(llm=llm)
        self.interaction_tool = interaction_tool
        self.memory = memory

    def __call__(self, task_description):
        """
        Generate intelligent plan based on context analysis

        Args:
            task_description: Task dict with user_id and item_id

        Returns:
            list: Optimized plan with dependencies and priorities
        """
        # 1. 快速上下文分析
        context = self._analyze_context(task_description)

        # 2. 从记忆中检索经验（如果可用）
        memory_guidance = ""
        if self.memory:
            try:
                memory_guidance = self.memory.retriveMemory(
                    f"planning for user {task_description['user_id']} item {task_description['item_id']}"
                )
            except:
                memory_guidance = ""

        # 3. 生成自适应计划
        prompt = f"""You are an intelligent planner for user behavior simulation tasks.

Task Context:
- User ID: {task_description['user_id']}
- Item ID: {task_description['item_id']}
- Scenario Type: {context['scenario']}
- Data Availability: User has {context['data_availability']['user_reviews']} reviews, Item has {context['data_availability']['item_reviews']} reviews

Past Successful Plans (if available):
{memory_guidance if memory_guidance else "No similar cases found - use standard approach"}

Generate an optimal plan considering:
1. **Data Dependencies**: What information needs to be collected first (e.g., user info before analyzing user patterns)
2. **Information Richness**: How much detail to gather based on data availability
3. **Efficiency**: Minimize unnecessary steps while ensuring quality
4. **Success Patterns**: Follow patterns from similar cases if available

Required steps (must include):
- Get user information (user profile, review history)
- Get business/item information (business details, attributes)
- Get business reviews (for context and similarity matching)
- Get user's historical reviews (for pattern analysis)

Output format - return a JSON list of steps:
[
    {{
        "description": "Clear step description",
        "reasoning_instruction": "What to think about or analyze in this step",
        "tool_use_instruction": {{"user_id": "...", "item_id": "..."}},
        "priority": "high"
    }}
]

Important:
- Use "high" priority for essential data collection steps
- Use "medium" priority for analysis steps
- Ensure logical order (collect data before analysis)
- Be specific about what data to extract and why
"""

        try:
            response = self.llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500,
            )

            plan = self._parse_plan(response)

            # 4. 验证和优化计划
            plan = self._validate_and_optimize(plan, context, task_description)

            self.plan = plan
            return plan

        except Exception as e:
            logging.warning(f"Smart planning failed, falling back to baseline: {e}")
            # 回退到简单计划
            return self._fallback_plan(task_description)

    def _analyze_context(self, task_description):
        """快速分析任务上下文"""
        if not self.interaction_tool:
            return {
                "scenario": "default",
                "data_availability": {"user_reviews": 0, "item_reviews": 0},
            }

        try:
            # 快速检查数据可用性（只获取数量，不加载完整数据）
            user_reviews = self.interaction_tool.get_reviews(
                user_id=task_description["user_id"]
            )
            item_reviews = self.interaction_tool.get_reviews(
                item_id=task_description["item_id"]
            )

            user_review_count = len(user_reviews) if user_reviews else 0
            item_review_count = len(item_reviews) if item_reviews else 0

            # 识别场景类型
            scenario = "default"
            if user_review_count < 3:
                scenario = "new_user"
            elif user_review_count > 50:
                scenario = "experienced_user"
            else:
                scenario = "regular_user"

            if item_review_count > 100:
                scenario += "_popular_item"
            elif item_review_count < 5:
                scenario += "_niche_item"
            else:
                scenario += "_standard_item"

            return {
                "scenario": scenario,
                "data_availability": {
                    "user_reviews": user_review_count,
                    "item_reviews": item_review_count,
                },
            }
        except Exception as e:
            logging.warning(f"Context analysis failed: {e}")
            return {
                "scenario": "default",
                "data_availability": {"user_reviews": 0, "item_reviews": 0},
            }

    def _parse_plan(self, response):
        """解析 LLM 输出的计划"""
        import json

        # 尝试提取 JSON
        # 方法1: 直接查找 JSON 数组
        json_match = re.search(r"\[[\s\S]*?\]", response)
        if json_match:
            try:
                json_str = json_match.group()
                plan = json.loads(json_str)
                if isinstance(plan, list) and len(plan) > 0:
                    return plan
            except json.JSONDecodeError:
                pass

        # 方法2: 查找多个 JSON 对象
        dict_strings = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response)
        if dict_strings:
            try:
                dicts = []
                for ds in dict_strings:
                    try:
                        d = json.loads(ds)
                        if isinstance(d, dict):
                            dicts.append(d)
                    except:
                        try:
                            d = ast.literal_eval(ds)
                            if isinstance(d, dict):
                                dicts.append(d)
                        except:
                            pass
                if dicts:
                    return dicts
            except:
                pass

        # 方法3: 回退到简单的字典解析
        dict_strings = re.findall(r"\{[^{}]*\}", response)
        if dict_strings:
            try:
                dicts = [ast.literal_eval(ds) for ds in dict_strings]
                return dicts
            except:
                pass

        # 如果都失败了，返回空列表
        logging.warning("Failed to parse plan from LLM response")
        return []

    def _validate_and_optimize(self, plan, context, task_description):
        """验证和优化计划"""
        if not plan:
            return self._fallback_plan(task_description)

        # 确保必要的步骤存在
        required_keywords = {
            "user": ["user", "profile", "history"],
            "item": ["item", "business", "product"],
            "reviews": ["review", "comment", "rating"],
        }

        has_user_step = any(
            any(
                keyword in step.get("description", "").lower()
                for keyword in required_keywords["user"]
            )
            for step in plan
        )
        has_item_step = any(
            any(
                keyword in step.get("description", "").lower()
                for keyword in required_keywords["item"]
            )
            for step in plan
        )
        has_reviews_step = any(
            any(
                keyword in step.get("description", "").lower()
                for keyword in required_keywords["reviews"]
            )
            for step in plan
        )

        # 如果缺少必要步骤，添加默认步骤
        if not has_user_step:
            plan.insert(
                0,
                {
                    "description": "Get user information and profile",
                    "reasoning instruction": "Collect user profile and review history",
                    "tool use instruction": {"user_id": task_description["user_id"]},
                    "priority": "high",
                },
            )

        if not has_item_step:
            plan.insert(
                1 if has_user_step else 0,
                {
                    "description": "Get business/item information",
                    "reasoning instruction": "Collect business details and attributes",
                    "tool use instruction": {"item_id": task_description["item_id"]},
                    "priority": "high",
                },
            )

        if not has_reviews_step:
            plan.append(
                {
                    "description": "Get business reviews for context",
                    "reasoning instruction": "Collect reviews from other users for similarity matching",
                    "tool use instruction": {"item_id": task_description["item_id"]},
                    "priority": "high",
                }
            )

        # 根据优先级排序（high -> medium -> low）
        priority_order = {"high": 0, "medium": 1, "low": 2}
        plan.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 1))

        # 确保 tool_use_instruction 格式正确
        for step in plan:
            if "tool use instruction" in step:
                tool_inst = step["tool use instruction"]
                if isinstance(tool_inst, dict):
                    # 确保包含必要的 ID
                    if (
                        "user_id" not in tool_inst
                        and "user" in step.get("description", "").lower()
                    ):
                        tool_inst["user_id"] = task_description["user_id"]
                    if "item_id" not in tool_inst and (
                        "item" in step.get("description", "").lower()
                        or "business" in step.get("description", "").lower()
                    ):
                        tool_inst["item_id"] = task_description["item_id"]

        return plan

    def _fallback_plan(self, task_description):
        """回退计划（当智能规划失败时）"""
        return [
            {
                "description": "First I need to find user information",
                "reasoning instruction": "Collect user profile and review history",
                "tool use instruction": {"user_id": task_description["user_id"]},
                "priority": "high",
            },
            {
                "description": "Next, I need to find business information",
                "reasoning instruction": "Collect business details and attributes",
                "tool use instruction": {"item_id": task_description["item_id"]},
                "priority": "high",
            },
            {
                "description": "Get business reviews for context",
                "reasoning instruction": "Collect reviews from other users",
                "tool use instruction": {"item_id": task_description["item_id"]},
                "priority": "high",
            },
        ]


class PlanningBaseline(PlanningBase):
    """Inherit from PlanningBase - kept for backward compatibility"""

    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)

    def __call__(self, task_description):
        """Override the parent class's __call__ method"""
        self.plan = [
            {
                "description": "First I need to find user information",
                "reasoning instruction": "None",
                "tool use instruction": {task_description["user_id"]},
            },
            {
                "description": "Next, I need to find business information",
                "reasoning instruction": "None",
                "tool use instruction": {task_description["item_id"]},
            },
        ]
        return self.plan


class ReasoningCOTEnhanced(ReasoningBase):
    """增强版 CoT 推理：针对用户行为模拟任务的链式思考"""

    def __init__(self, profile_type_prompt, llm, memory=None):
        """Initialize the enhanced CoT reasoning module"""
        super().__init__(
            profile_type_prompt=profile_type_prompt, memory=memory, llm=llm
        )

    def __call__(self, task_description: str, feedback: str = ""):
        """
        Enhanced Chain of Thought reasoning for user behavior simulation

        Args:
            task_description: Full task description with user info, business info, etc.
            feedback: Optional feedback from previous attempts
        """
        # 构建结构化的 CoT prompt
        prompt = f"""You are simulating a real user's behavior on Yelp. You need to think step by step to generate an authentic rating and review.

{task_description}

Now, think through this task step by step:

**Step 1: Analyze User Profile and History**
- Review the user's profile information and past review history
- Identify the user's rating patterns (e.g., tends to give high/low ratings, average rating)
- Note the user's review style (length, tone, focus areas)
- Identify what aspects the user typically cares about (service, food quality, ambiance, etc.)

**Step 2: Analyze Business Information**
- Review the business details, attributes, and characteristics
- Note the business category, average rating, and key features
- Identify what makes this business stand out (positive or negative aspects)

**Step 3: Consider Similar Reviews**
- Review what other users have said about this business
- Identify common themes and concerns
- Note how similar users rated this business

**Step 4: Match User Preferences with Business Characteristics**
- Compare the user's historical preferences with this business's features
- Identify alignment or misalignment points
- Consider how this business would fit the user's typical preferences

**Step 5: Predict Rating**
- Based on the analysis above, predict what rating (1.0-5.0) this user would give
- Consider:
  * User's historical rating patterns
  * How well the business matches user preferences
  * Business quality indicators (average rating, attributes)
  * Similarity to businesses the user has reviewed before
- Justify your rating prediction with specific reasoning

**Step 6: Generate Review Text**
- Write a review that matches the user's historical review style
- Focus on aspects the user typically mentions in their reviews
- Use similar tone, length, and detail level as the user's past reviews
- Incorporate specific details about this business
- Make it authentic and personal, as if written by this specific user

**Step 7: Finalize Output**
- Ensure the rating is one of: 1.0, 2.0, 3.0, 4.0, 5.0
- Ensure the review is 2-4 sentences and matches user's style
- Format your final answer as:
  stars: [rating]
  review: [review text]

Now, provide your step-by-step reasoning and final answer:"""

        messages = [{"role": "user", "content": prompt}]

        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,  # Slightly higher for more creative reasoning
            max_tokens=2000,  # More tokens for detailed reasoning
        )

        return reasoning_result


class ReasoningCOTVoting(ReasoningBase):
    """CoT + 投票机制：生成多个推理结果并通过投票选择最佳答案"""

    def __init__(
        self, profile_type_prompt, llm, memory=None, num_candidates=5, num_votes=0
    ):
        """
        Initialize CoT voting reasoning module

        Args:
            profile_type_prompt: Profile type prompt
            llm: LLM instance
            memory: Memory module (optional)
            num_candidates: Number of candidate answers to generate (default: 5)
            num_votes: Number of votes to cast for selecting best answer (default: 5)
        """
        super().__init__(
            profile_type_prompt=profile_type_prompt, memory=memory, llm=llm
        )
        self.num_candidates = num_candidates
        self.num_votes = num_votes

    def __call__(self, task_description: str, feedback: str = ""):
        """
        Generate multiple CoT reasoning results and select best via voting

        Args:
            task_description: Full task description with user info, business info, etc.
            feedback: Optional feedback from previous attempts
        """
        # 构建结构化的 CoT prompt
        cot_prompt = f"""You are simulating a real user's behavior on Yelp. You need to think step by step to generate an authentic rating and review.

{task_description}

Now, think through this task step by step:

**Step 1: Analyze User Profile and History**
- Review the user's profile information and past review history
- Identify the user's rating patterns (e.g., tends to give high/low ratings, average rating)
- Note the user's review style (length, tone, focus areas)
- Identify what aspects the user typically cares about (service, food quality, ambiance, etc.)

**Step 2: Analyze Business Information**
- Review the business details, attributes, and characteristics
- Note the business category, average rating, and key features
- Identify what makes this business stand out (positive or negative aspects)

**Step 3: Consider Similar Reviews**
- Review what other users have said about this business
- Identify common themes and concerns
- Note how similar users rated this business

**Step 4: Match User Preferences with Business Characteristics**
- Compare the user's historical preferences with this business's features
- Identify alignment or misalignment points
- Consider how this business would fit the user's typical preferences

**Step 5: Predict Rating**
- Based on the analysis above, predict what rating (1.0-5.0) this user would give
- Consider:
  * User's historical rating patterns
  * How well the business matches user preferences
  * Business quality indicators (average rating, attributes)
  * Similarity to businesses the user has reviewed before
- Justify your rating prediction with specific reasoning

**Step 6: Generate Review Text**
- Write a review that matches the user's historical review style
- Focus on aspects the user typically mentions in their reviews
- Use similar tone, length, and detail level as the user's past reviews
- Incorporate specific details about this business
- Make it authentic and personal, as if written by this specific user

**Step 7: Finalize Output**
- Ensure the rating is one of: 1.0, 2.0, 3.0, 4.0, 5.0
- Ensure the review is 2-4 sentences and matches user's style
- Format your final answer as:
  stars: [rating]
  review: [review text]

Now, provide your step-by-step reasoning and final answer:"""

        messages = [{"role": "user", "content": cot_prompt}]

        # Step 1: Generate multiple candidate answers
        logging.info(f"Generating {self.num_candidates} candidate answers...")
        candidate_results = self.llm(
            messages=messages,
            temperature=0.3,  # Higher temperature for diversity
            max_tokens=2000,
            n=self.num_candidates,  # Generate multiple candidates
        )

        # Ensure we have a list
        if not isinstance(candidate_results, list):
            candidate_results = [candidate_results]

        # Step 2: Extract ratings and reviews from candidates
        candidates = []
        for i, result in enumerate(candidate_results):
            try:
                stars, review = self._extract_stars_and_review(result)
                if stars is not None and review:
                    candidates.append(
                        {
                            "id": i + 1,
                            "stars": stars,
                            "review": review,
                            "full_response": result,
                        }
                    )
            except Exception as e:
                logging.warning(f"Failed to parse candidate {i+1}: {e}")
                continue

        if not candidates:
            logging.warning("No valid candidates generated, using first result")
            return candidate_results[0] if candidate_results else ""

        # Step 3: If only one candidate, return it
        if len(candidates) == 1:
            return candidates[0]["full_response"]

        #         # Step 4: Voting phase - ask LLM to vote for best answer
        #         voting_prompt = f"""You are evaluating multiple candidate answers for a user behavior simulation task.

        # Task Context:
        # {task_description}

        # You have {len(candidates)} candidate answers. Each candidate provides:
        # 1. A star rating (1.0-5.0)
        # 2. A review text

        # Evaluate each candidate based on:
        # - **Rating Accuracy**: Does the rating match the user's historical patterns and the business quality?
        # - **Review Authenticity**: Does the review match the user's writing style and typical focus areas?
        # - **Consistency**: Are the rating and review consistent with each other?
        # - **Relevance**: Does the review address relevant aspects of the business?

        # Candidates:
        # """

        #         for candidate in candidates:
        #             voting_prompt += f"""
        # Candidate {candidate['id']}:
        # Rating: {candidate['stars']}
        # Review: {candidate['review']}
        # """

        #         voting_prompt += f"""
        # Your task: Select the BEST candidate that most accurately simulates this user's behavior.

        # Output format:
        # The best answer is [candidate_id]

        # Where [candidate_id] is the number (1-{len(candidates)}) of the candidate you think is best.
        # """

        #         # Generate votes
        #         vote_messages = [{"role": "user", "content": voting_prompt}]
        #         vote_outputs = self.llm(
        #             messages=vote_messages,
        #             temperature=0.2,  # Lower temperature for more consistent voting
        #             max_tokens=500,
        #             n=self.num_votes,  # Generate multiple votes
        #         )

        #         # Ensure we have a list
        #         if not isinstance(vote_outputs, list):
        #             vote_outputs = [vote_outputs]

        #         # Count votes
        #         vote_counts = {candidate["id"]: 0 for candidate in candidates}
        #         for vote_output in vote_outputs:
        #             candidate_id = self._parse_vote(vote_output, len(candidates))
        #             if candidate_id:
        #                 vote_counts[candidate_id] += 1

        #         # Select winner (candidate with most votes)
        #         winner_id = max(vote_counts, key=vote_counts.get)
        #         winner = next(c for c in candidates if c["id"] == winner_id)

        #         logging.info(f"Voting results: {vote_counts}, Winner: Candidate {winner_id}")

        # return winner["full_response"]
        stars_array = np.array([c["stars"] for c in candidates], dtype=float)
        median_star = float(np.median(stars_array))

        winner = min(candidates, key=lambda c: abs(float(c["stars"]) - median_star))

        logging.info(
            f"Selected candidate with stars {winner['stars']} (median {median_star})"
        )

        return winner["full_response"]

    def _extract_stars_and_review(self, text):
        """Extract stars and review from LLM response"""
        try:
            # Try to find stars: line
            stars_line = None
            review_line = None

            for line in text.split("\n"):
                if "stars:" in line.lower():
                    stars_line = line
                if "review:" in line.lower():
                    review_line = line

            if not stars_line or not review_line:
                # Try alternative formats
                stars_match = re.search(r"stars?[:\s]+([0-9.]+)", text, re.IGNORECASE)
                review_match = re.search(
                    r"review[:\s]+(.+?)(?=\n\n|\nstars?|$)",
                    text,
                    re.IGNORECASE | re.DOTALL,
                )

                if stars_match:
                    stars = float(stars_match.group(1))
                else:
                    return None, None

                if review_match:
                    review = review_match.group(1).strip()
                else:
                    return None, None
            else:
                stars = float(stars_line.split(":")[1].strip())
                review = review_line.split(":")[1].strip()

            # Validate stars
            if stars < 1.0 or stars > 5.0:
                return None, None

            return stars, review

        except Exception as e:
            logging.warning(f"Error extracting stars and review: {e}")
            return None, None

    def _parse_vote(self, vote_text, max_candidate_id):
        """Parse vote from LLM output"""
        # Try to find "best answer is X" or "candidate X" pattern
        patterns = [
            r"best answer is\s*(\d+)",
            r"candidate\s*(\d+)",
            r"answer\s*(\d+)",
            r"(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, vote_text, re.IGNORECASE)
            if match:
                candidate_id = int(match.group(1))
                if 1 <= candidate_id <= max_candidate_id:
                    return candidate_id

        return None


class ReasoningBaseline(ReasoningBase):
    """Inherit from ReasoningBase - kept for backward compatibility"""

    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = """
{task_description}"""
        prompt = prompt.format(task_description=task_description)

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(messages=messages, temperature=0.0, max_tokens=1000)

        return reasoning_result


class MySimulationAgent(SimulationAgent):
    """Participant's implementation of SimulationAgent."""

    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgent"""
        super().__init__(llm=llm)
        # Initialize memory first (needed for smart planning)
        self.memory = MemoryDILU(llm=self.llm)
        # Initialize smart planning with interaction_tool (will be set later)
        # Note: interaction_tool is set by Simulator, so we pass None initially
        self.planning = PlanningSmart(
            llm=self.llm, interaction_tool=None, memory=self.memory
        )
        # Use CoT + Voting reasoning for better accuracy and consistency
        # num_candidates: 生成m个候选答案, num_votes: 进行n轮投票
        self.reasoning = ReasoningCOTVoting(
            profile_type_prompt="",
            llm=self.llm,
            memory=self.memory,
            num_candidates=5,
            num_votes=0,
        )

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            dict: {"stars": float, "review": str}
        """
        try:
            # Update planning module with interaction_tool (now available)
            if hasattr(self, "interaction_tool") and self.interaction_tool:
                self.planning.interaction_tool = self.interaction_tool

            plan = self.planning(task_description=self.task)

            # Initialize variables
            user = None
            business = None

            # Execute plan steps
            for sub_task in plan:
                description = sub_task.get("description", "").lower()
                tool_instruction = sub_task.get(
                    "tool use instruction", {}
                ) or sub_task.get("tool_use_instruction", {})

                # Handle user information step
                if "user" in description or "profile" in description:
                    if (
                        isinstance(tool_instruction, dict)
                        and "user_id" in tool_instruction
                    ):
                        user_id = tool_instruction["user_id"]
                    else:
                        user_id = self.task["user_id"]
                    user = str(self.interaction_tool.get_user(user_id=user_id))

                # Handle business/item information step
                elif (
                    "business" in description
                    or "item" in description
                    or "product" in description
                ):
                    if (
                        isinstance(tool_instruction, dict)
                        and "item_id" in tool_instruction
                    ):
                        item_id = tool_instruction["item_id"]
                    else:
                        item_id = self.task["item_id"]
                    business = str(self.interaction_tool.get_item(item_id=item_id))

            # Ensure we have user and business info (fallback if plan didn't include them)
            if user is None:
                user = str(self.interaction_tool.get_user(user_id=self.task["user_id"]))
            if business is None:
                business = str(
                    self.interaction_tool.get_item(item_id=self.task["item_id"])
                )
            reviews_item = self.interaction_tool.get_reviews(
                item_id=self.task["item_id"]
            )
            for review in reviews_item:
                review_text = review["text"]
                self.memory(f"review: {review_text}")
            reviews_user = self.interaction_tool.get_reviews(
                user_id=self.task["user_id"]
            )
            review_similar = self.memory(f'{reviews_user[0]["text"]}')

            # Get item reviews + store in memory
            reviews_item = self.interaction_tool.get_reviews(
                item_id=self.task["item_id"]
            )
            for review in reviews_item:
                review_text = review["text"]
                self.memory(f"review: {review_text}")

            # Compute item mean rating
            item_ratings = [r["stars"] for r in reviews_item] if reviews_item else []
            if item_ratings:
                item_mean = sum(item_ratings) / len(item_ratings)
                item_count = len(item_ratings)
            else:
                item_mean = 3.5
                item_count = 0

            # Get user reviews
            reviews_user = self.interaction_tool.get_reviews(
                user_id=self.task["user_id"]
            )

            # Compute user mean rating
            user_ratings = [r["stars"] for r in reviews_user] if reviews_user else []
            if user_ratings:
                user_mean = sum(user_ratings) / len(user_ratings)
            else:
                user_mean = 3.5

            # Use one of the user's real reviews as a seed for memory
            if reviews_user:
                review_similar = self.memory(f'{reviews_user[0]["text"]}')
            else:
                review_similar = ""

            task_description = f"""
            You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

            You need to write a review for this business: {business}
            
            Historical behavior stats:
            - Your average star rating across past reviews: {user_mean:.2f}
            - This business's average rating from other users: {item_mean:.2f} based on {item_count} reviews

            Others have reviewed this business before: {review_similar}

            Please analyze the following aspects carefully:
            1. Based on your user profile and review style, what rating would you give this business? Remember that many users give 5-star ratings for excellent experiences that exceed expectations, and 1-star ratings for very poor experiences that fail to meet basic standards.
            2. Given the business details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this business stand out or negative aspects that severely impact the experience.
            3. Consider how other users might engage with your review in terms of:
            - Useful: How informative and helpful is your review?
            - Funny: Does your review have any humorous or entertaining elements?
            - Cool: Is your review particularly insightful or praiseworthy?

            Requirements:
            - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
            - If the business meets or exceeds expectations in key areas, consider giving a 5-star rating
            - If the business fails significantly in key areas, consider giving a 1-star rating
            - Review text should be 2-4 sentences, focusing on your personal experience and emotional response
            - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
            - Maintain consistency with your historical review style and rating patterns
            - Focus on specific details about the business rather than generic comments
            - Be generous with ratings when businesses deliver quality service and products
            - Be critical when businesses fail to meet basic standards

            Format your response exactly as follows:
            stars: [your rating]
            review: [your review]
            """
            result = self.reasoning(task_description)

            try:
                stars_line = [line for line in result.split("\n") if "stars:" in line][
                    0
                ]
                review_line = [
                    line for line in result.split("\n") if "review:" in line
                ][0]
            except:
                print("Error:", result)

            stars = float(stars_line.split(":")[1].strip())
            review_text = review_line.split(":")[1].strip()

            if len(review_text) > 512:
                review_text = review_text[:512]

            return {"stars": stars, "review": review_text}
        except Exception as e:
            print(f"Error in workflow: {e}")
            return {"stars": 0, "review": ""}


if __name__ == "__main__":
    # Set the data
    task_set = "yelp"  # "goodreads" or "yelp"
    simulator = Simulator(
        data_dir="/Users/anniezhou/Desktop/245 Project/AgentSocietyChallenge/dataset",
        device="gpu",
        cache=False,
    )
    simulator.set_task_and_groundtruth(
        task_dir=f"/Users/anniezhou/Desktop/245 Project/AgentSocietyChallenge/example/track1/{task_set}/tasks",
        groundtruth_dir=f"/Users/anniezhou/Desktop/245 Project/AgentSocietyChallenge/example/track1/{task_set}/groundtruth",
    )

    import os

    # Set the agent and LLM
    simulator.set_agent(MySimulationAgent)
    simulator.set_llm(OpenAILLM(api_key=os.environ["API_KEY"]))

    # Run the simulation
    # If you don't set the number of tasks, the simulator will run all tasks.
    # outputs = simulator.run_simulation(
    #     number_of_tasks=50, enable_threading=True, max_workers=10
    # )

    # # Evaluate the agent
    # evaluation_results = simulator.evaluate()
    # with open(
    #     f"/Users/anniezhou/Desktop/245 Project/AgentSocietyChallenge/Evaluation/evaluation_results_{task_set}_50tasks.json",
    #     "w",
    # ) as f:
    #     json.dump(evaluation_results, f, indent=4)
    for n_tasks in [10, 50, 200]:
        print(f"\n===== Running simulation with {n_tasks} tasks =====")

        outputs = simulator.run_simulation(
            number_of_tasks=n_tasks,
            enable_threading=True,
            max_workers=10,
        )

        evaluation_results = simulator.evaluate()

        out_path = (
            f"/Users/anniezhou/Desktop/245 Project/AgentSocietyChallenge/"
            f"Evaluation/evaluation_results_new_{n_tasks}tasks.json"
        )
        with open(out_path, "w") as f:
            json.dump(evaluation_results, f, indent=4)

        print(f"Saved evaluation to {out_path}")

    # Get evaluation history
    evaluation_history = simulator.get_evaluation_history()
