import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
    
        #at init load tokens from disk ans store them in memory
        """Read the input data"""
        with open('input.txt', 'r') as f:
            text = f.read()

        """Get the gpt2 encodings from tiktoken and encode the entire dataset"""
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // self.B*self.T} batches")

        #state
        self.current_positon = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_positon : self.current_positon + B*T +1]
        buf = buf.to(device)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        #advance the position in the tensor
        self.current_positon+= B*T
        if(self.current_positon + (B*T +1) > len(self.tokens)):
            self.current_positon = 0
        return x, y