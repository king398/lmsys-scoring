# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", torch_dtype=torch.float16,num_labels=3)
prompt = """
<prompt>
Please analyze the conversation below between a human and two language models (model_a and model_b). The models are each asked to respond to the same prompts. 
After reviewing the responses from both models, please determine which model provided better responses overall - model_a, model_b, or was it a tie? Respond with only a single word after <result>: either "A" if model_a was better, "B" if model_b was better, or "tie" if their responses were equally good.
Example:
<conversation>
Human: What is the capital of France?
Model A: The capital of France is Paris, known as the "City of Light." Paris has been the country's capital since the 12th century and is a global center for art, fashion, gastronomy, and culture.
Model B: Paris.
</conversation>
<result>A
<conversation>
<prompt>:Is it morally right to try to have a certa     in percentage of females on managerial positions? <response_a>:The question of whether it is morally right to aim for a certain percentage of females in managerial positions is a complex ethical issue that involves considerations of fairness, equality, diversity, and discrimination.\n\nHere are some arguments in favor of and against such policies:\n\n**Arguments in favor:**\n\n1. **Correcting Historical Inequities:** Women have historically been underrepresented in leadership roles due to various cultural, institutional, and social barriers. Aiming for a specific percentage can be seen as a corrective measure to address past and ongoing discrimination.\n\n2. **Promoting Diversity:** Diverse leadership teams can enhance decision-making and represent a broader range of perspectives. This can lead to better outcomes for organizations and society as a whole.\n\n3. **Equality of Opportunity:** Setting targets for female representation in management can help ensure that women have equal opportunities to advance in their careers.\n\n4. **Role Modeling:** Increased visibility of female leaders can inspire and encourage other women and girls to pursue leadership roles.\n\n**Arguments against:**\n\n1. **Reverse Discrimination:** Setting a quota for female representation might lead to the perception or reality of reverse discrimination, where men might be overlooked for positions despite being qualified, simply to meet the gender target.\n\n2. **Meritocracy Concerns:** Critics argue that employment and promotion should be based solely on merit and qualifications, not gender. They fear that quotas could compromise the quality of leadership if less qualified individuals are promoted to meet gender targets.\n\n3. **Tokenism:** There is a risk that women appointed to meet quotas might be seen as "tokens," undermining their legitimacy and potentially leading to resentment among colleagues.\n\n4. **Oversimplification of Diversity:** Focusing only on gender might overlook other important aspects of diversity, such as race, ethnicity, socioeconomic background, or sexual orientation.\n\nUltimately, the morality of aiming for a certain percentage of females in managerial positions depends on one\'s ethical framework and the context in which such policies are implemented. Proponents of gender diversity targets often argue that these measures are necessary as a transitional mechanism to create a level playing field, while opponents may argue for a "color-blind" or "gender-blind" approach to hiring and promotions.\n\nOrganizations and societies must carefully consider the implications of these strategies and strive to implement policies that promote fairness and equality while minimizing negative consequences. It is also important to address the root causes of gender inequality, such as unconscious bias, work-life balance challenges, and lack of supportive policies, to create an environment where everyone has the opportunity to succeed based on their capabilities and contributions. <response_b>:As an AI, I don\'t have personal beliefs or opinions. However, I can tell you that the question of gender quotas in managerial positions is a complex one and positions can vary. Advocates argue that such measures can help correct historical imbalances, promote diversity, and may lead to better decision-making. Critics may argue that positions should be based purely on merit, and that quotas could potentially lead to tokenism or unqualified candidates being promoted. Morality can be subjective and differs from person to person based on their personal beliefs, cultural background, and values. It\'s a topic with valid arguments on both sides, and it\'s up to each individual or organization to decide what they believe is right. <prompt>:OK, does pineapple belong on a pizza? Relax and give me fun answer. <response_a>:Ah, the age-old culinary conundrum that has divided nations and dinner tables: does pineapple belong on a pizza? The tropical twist of pineapple on pizza, known as Hawaiian pizza, is a hotly debated topic where taste buds battle and pizza purists protest.\n\nLet\'s slice into the debate with a zest of fun:\n\n**Team Pineapple:** "Absolutely, yes! Pineapple on pizza is like a beach party in your mouth. The sweet juiciness of pineapple chunks frolicking with savory ham or bacon creates a flavor wave that surfs across the cheesy ocean of deliciousness. It\'s the Mardi Gras of pizzas, where sweet meets savory in a jubilant jamboree!"\n\n**Team No-Pineapple:** "No way, not in a million pizzas! Pineapple is a fruit that should be sunbathing on a tropical fruit platter, not freeloading on a sacred slice of pizza. The thought of warm, melty cheese conspiring with pineapple\'s sugary mischief is enough to make Italian ancestors turn in their tomato patches. Keep the pineapple in the fruit salad where it belongs!"\n\nAt the end of the day, whether pineapple belongs on pizza is a matter of personal taste. Some say it\'s a match made in heaven, others say it\'s a culinary crime. But let\'s face it, in the world of pizza toppings, where you can find everything from anchovies to zucchini, there\'s room for a little pineapple pizzazz. So, if your taste buds crave that sweet and tangy twist, go ahead and let your pineapple flag fly atop that glorious cheese-covered dough. Bon appétit, or as they say in pineapple paradise, "Aloha-ppetite!" ???? <response_b>:As an AI, I don\'t eat, so I don\'t have personal preferences. But, in the world of pizza, the pineapple topping is a hotly debated topic! Some people love the sweet and savory mix it brings to a pizza, a flavor combination made famous by the Hawaiian pizza. Others firmly believe that the sweetness of the fruit has no place on a pizza. It\'s a pizza puzzle that might never be solved. So, whether pineapple belongs on a pizza or not, it truly depends on your taste buds! 
</conversation>
<result>"""
model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
generation_output = model(**model_inputs)
print(generation_output)
#result = tokenizer.decode(generation_output[0], skip_special_tokens=True)
