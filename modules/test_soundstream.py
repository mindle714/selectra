import torch
import torchaudio
import soundstream
import soundfile

waveform, _ = soundfile.read('x.wav')
waveform = torch.from_numpy(waveform).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
audio_codec = soundstream.load_codec()  # downloads model from Hugging Face
audio_codec.eval()

quantized = audio_codec(waveform, mode='encode')
recovered = audio_codec(quantized, mode='decode')
torchaudio.save('x_test2.wav', recovered[0], 16000)

e = audio_codec.encoder(waveform)
res = audio_codec.quantizer(e.permute((0,2,1)))
q2 = audio_codec.quantizer.get_output_from_indices(res[1])
recovered2 = audio_codec(quantized, mode='decode')
assert torch.all(recovered == recovered2)

torchaudio.save('x_test3.wav', recovered2[0], 16000)
print(res)