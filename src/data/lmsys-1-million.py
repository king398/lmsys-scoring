from datasets import load_dataset

ds = load_dataset("lmsys/lmsys-chat-1m")
df  = ds['train'].to_pandas()

print(df['content'])
def prepare_text(group):
    text = """"""
    for i,content in enumerate(group['conversation']):
        print(content)

    raise NotImplementedError

df.groupby("conversation_id").apply(prepare_text)