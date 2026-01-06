# Xây dựng prompt chi tiết
                system_instruction = (
                    "You are a Senior Research Scientist at a top-tier AI lab. Your task is to synthesize "
                    "research limitations and identify high-impact, novel research opportunities."
                )

                user_prompt = f"""
                I will provide you with the 'Limitations' and 'Future Work' sections extracted from several recent papers 
                from ICLR 2026 on the topic: '{topic}'.

                Based on this data, please conduct a deep architectural and theoretical analysis to generate a 
                Research Proposal Report. Your response must be in English and follow this structure:

                ### 1. Critical Synthesis of Existing Gaps
                - Identify the top 3 most critical unsolved problems that are common across these papers.
                - Explain *why* these remain unsolved despite current state-of-the-art efforts.

                ### 2. Novel Research Directions & Proposed Methods
                For each of the 3 problems identified above, propose a specific, NOVEL research direction. 
                For each direction, include:
                - **Concept:** A clear statement of the new idea.
                - **Proposed Method:** Describe a technical approach or architecture change (e.g., a new loss function, 
                a hybrid transformer-state space model, a novel regularization technique, etc.) to solve the gap. 
                Ensure the method demonstrates high NOVELTY and goes beyond incremental improvements.
                - **Expected Impact:** How this method changes the current paradigm.

                ### 3. Feasibility & Evaluation
                - Suggest a specific dataset or evaluation metric that should be used to validate these novel methods.

                ---
                RAW DATA FROM ICLR 2026 PAPERS:
                {full_context}
                """
