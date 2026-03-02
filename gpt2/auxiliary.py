def generate(text, my_model=None, max_tokens=d_context, temperature=0.7, top_k=40):
  assert my_model is not None, "my_model is None"
  input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
  my_model.eval()
  with torch.no_grad():
    for _ in range(max_tokens):
      if input_ids.shape[1] >= d_context:
        print(f'BREAK: require too much tokens')
        break
      logits = my_model(input_ids)
      logits = logits[:,-1,:]
      logits = logits/temperature
      if top_k > 0:
        # 找到第 k 大的那个值
        v, _ = torch.topk(logits, top_k)
        min_val = v[:, -1].unsqueeze(1)
        # 将小于第 k 大值的 logits 设为负无穷，这样 softmax 后概率为 0
        logits = torch.where(
            logits < min_val,
            torch.tensor(float('-inf')).to(device),
            logits
        )
      probs = F.softmax(logits, dim=-1)
      newidx = torch.multinomial(probs, num_samples=1)
      input_ids = torch.cat((input_ids, newidx), dim=1)
    print(f'{tokenizer.decode(input_ids[0].tolist())}')