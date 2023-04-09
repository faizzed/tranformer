from generator import Generator

model = Generator()
sentence = 'On its day'
for i in range(5):
    out = model.generate(sentence)
    sentence += f' {out}'

print(sentence)