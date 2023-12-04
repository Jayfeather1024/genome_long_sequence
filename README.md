# Setting Up Environment
Follow `setup.sh` step by step to create virtual environments. Two conda environments are needed, one for encoder and decoder training and the other for diffusion model training. 

# Dataset
Download the dataset into the folder `data`.

# Pretrained GenSLM
Download the pretrained GenSLM into the folder `genslm_model`.

# Getting the Latent Features
```
from get_latent_features import load_encoder, load_tokenizer, get_latent_feature

sequence = "ACG GGC CCG AGC AAA"
model_path = "/eagle/projects/CVD-Mol-AI/for_yuntian/genome_results/encoder_genslm_28289_final/run_l0.0005_b32_cbow_diff/checkpoints/epoch=2-step=18746.ckpt"
tokenizer_path = "/eagle/projects/CVD-Mol-AI/for_yuntian/genome_model/codon_wordlevel_100vocab_added.json"

tokenizer = load_tokenizer(tokenizer_path)
model = load_encoder(model_path)
latent_feature = get_latent_feature(sequence, model, tokenizer)
print(latent_feature)
```
# Getting Logits
```
import torch
from get_logits import load_encoder, load_decoder, get_logits

# file path
feature_file = "/lus/eagle/projects/CVD-Mol-AI/for_yuntian/genome_results/encoder_genslm_28289_final/run_l0.0005_b32_cbow_diff/diffusion/seed_81482_lr0.00002/sample_21395_repaint/zs.pt.masked.seeds.1234"
encoder_path = "/lus/eagle/projects/CVD-Mol-AI/for_yuntian/genome_results/encoder_genslm_28289_final/run_l0.0005_b32_cbow_diff/checkpoints/epoch=2-step=18746.ckpt"
decoder_path = "/lus/eagle/projects/CVD-Mol-AI/for_yuntian//genome_results/encoder_genslm_28289_final/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_full/pytorch_model.bin"
tokenizer_path = "/eagle/projects/CVD-Mol-AI/for_yuntian/genome_model/codon_wordlevel_100vocab_added.json"

# load features
features = torch.load(feature_file)
previous_features = features[0]['padded_sentence_embeddings'][0]
current_features = features[0]['padded_sentence_embeddings'][1]

# genome with 512 codons
sequence = "ACC AAC CAA CTT TCG ATC TCT TGT AGA TCT GTT CTC TAA ACG AAC TTT AAA ATC TGT GTG GCT GTC ACT CGG CTG CAT GCT TAG TGC ACT CAC GCA GTA TAA TTA ATA ACT AAT TAC TGT CGT TGA CAG GAC ACG AGT AAC TCG TCT ATC TTC TGC AGG CTG CTT ACG GTT TCG TCC GTT TTG CAG CCG ATC ATC AGC ACA TCT AGG TTT TGT CCG GGT GTG ACC GAA AGG TAA GAT GGA GAG CCT TGT CCC TGG TTT CAA CGA GAA AAC ACA CGT CCA ACT CAG TTT GCC TGT TTT ACA GGT TCG CGA CGT GCT CGT ACG TGG CTT TGG AGA CTC CGT GGA GGA GGT CTT ATC AGA GGC ACG TCA ACA TCT TAA AGA TGG CAC TTG TGG CTT AGT AGA AGT TGA AAA AGG CGT TTT GCC TCA ACT TGA ACA GCC CTA TGT GTT CAT CAA ACG TTC GGA TGC TCG AAC TGC ACC TCA TGG TCA TGT TAT GGT TGA GCT GGT AGC AGA ACT CGA AGG CAT TCA GTA CGG TCG TAG TGG TGA GAC ACT TGG TGT CCT TGT CCC TCA TGT GGG CGA AAT ACC AGT GGC TTA CCG CAA GGT TCT TCT TCG TAA GAA CGG TAA TAA AGG AGC TGG TGG CCA TAG TTA CGG CGC CGA TCT AAA GTC ATT TGA CTT AGG CGA CGA GCT TGG CAC TGA TCC TTA TGA AGA TTT TCA AGA AAA CTG GAA CAC TAA ACA TAG CAG TGG TGT TAC CCG TGA ACT CAT GCG TGA GCT TAA CGG AGG GGC ATA CAC TCG CTA TGT CGA TAA CAA CTT CTG TGG CCC TGA TGG CTA CCC TCT TGA GTG CAT TAA AGA CCT TCT AGC ACG TGC TGG TAA AGC TTC ATG CAC TTT GTC CGA ACA ACT GGA CTT TAT TGA CAC TAA GAG GGG TGT ATA CTG CTG CCG TGA ACA TGA GCA TGA AAT TGC TTG GTA CAC GGA ACG TTC TGA AAA GAG CTA TGA ATT GCA GAC ACC TTT TGA AAT TAA ATT GGC AAA GAA ATT TGA CAC CTT CAA TGG GGA ATG TCC AAA TTT TGT ATT TCC CTT AAA TTC CAT AAT CAA GAC TAT TCA ACC AAG GGT TGA AAA GAA AAA GCT TGA TGG CTT TAT GGG TAG AAT TCG ATC TGT CTA TCC AGT TGC GTC ACC AAA TGA ATG CAA CCA AAT GTG CCT TTC AAC TCT CAT GAA GTG TGA TCA TTG TGG TGA AAC TTC ATG GCA GAC GGG CGA TTT TGT TAA AGC CAC TTG CGA ATT TTG TGG CAC TGA GAA TTT GAC TAA AGA AGG TGC CAC TAC TTG TGG TTA CTT ACC CCA AAA TGC TGT TGT TAA AAT TTA TTG TCC AGC ATG TCA CAA TTC AGA AGT AGG ACC TGA GCA TAG TCT TGC CGA ATA CCA TAA TGA ATC TGG CTT GAA AAC CAT TCT TCG TAA GGG TGG TCG CAC TAT TGC CTT TGG AGG CTG TGT GTT CTC TTA TGT TGG TTG CCA TAA CAA GTG TGC CTA TTG GGT TCC ACG TGC TAG CGC TAA CAT AGG TTG TAA"

# load model
tokenizer, model = load_decoder(encoder_path, decoder_path, tokenizer_path)

# generate logits
logits = get_logits(sequence, model, tokenizer, previous_features, current_features)
print(logits)
```

# Sequence Generation
```
import torch
from get_sequence import load_encoder, load_decoder, generate_sequence

# file path
feature_file = "/lus/eagle/projects/CVD-Mol-AI/for_yuntian/genome_results/encoder_genslm_28289_final/run_l0.0005_b32_cbow_diff/diffusion/seed_81482_lr0.00002/sample_21395_repaint/zs.pt.masked.seeds.1234"
encoder_path = "/lus/eagle/projects/CVD-Mol-AI/for_yuntian/genome_results/encoder_genslm_28289_final/run_l0.0005_b32_cbow_diff/checkpoints/epoch=2-step=18746.ckpt"
decoder_path = "/lus/eagle/projects/CVD-Mol-AI/for_yuntian//genome_results/encoder_genslm_28289_final/run_l0.0005_b32_cbow_diff/decoder_23434/decoder_gpt2_e5_full/pytorch_model.bin"
tokenizer_path = "/eagle/projects/CVD-Mol-AI/for_yuntian/genome_model/codon_wordlevel_100vocab_added.json"

# load features
features = torch.load(feature_file)
previous_features = features[0]['padded_sentence_embeddings'][0]
current_features = features[0]['padded_sentence_embeddings'][1]

# genome with 512 codons
sequence = "ACC AAC CAA CTT TCG ATC TCT TGT AGA TCT GTT CTC TAA ACG AAC TTT AAA ATC TGT GTG GCT GTC ACT CGG CTG CAT GCT TAG TGC ACT CAC GCA GTA TAA TTA ATA ACT AAT TAC TGT CGT TGA CAG GAC ACG AGT AAC TCG TCT ATC TTC TGC AGG CTG CTT ACG GTT TCG TCC GTT TTG CAG CCG ATC ATC AGC ACA TCT AGG TTT TGT CCG GGT GTG ACC GAA AGG TAA GAT GGA GAG CCT TGT CCC TGG TTT CAA CGA GAA AAC ACA CGT CCA ACT CAG TTT GCC TGT TTT ACA GGT TCG CGA CGT GCT CGT ACG TGG CTT TGG AGA CTC CGT GGA GGA GGT CTT ATC AGA GGC ACG TCA ACA TCT TAA AGA TGG CAC TTG TGG CTT AGT AGA AGT TGA AAA AGG CGT TTT GCC TCA ACT TGA ACA GCC CTA TGT GTT CAT CAA ACG TTC GGA TGC TCG AAC TGC ACC TCA TGG TCA TGT TAT GGT TGA GCT GGT AGC AGA ACT CGA AGG CAT TCA GTA CGG TCG TAG TGG TGA GAC ACT TGG TGT CCT TGT CCC TCA TGT GGG CGA AAT ACC AGT GGC TTA CCG CAA GGT TCT TCT TCG TAA GAA CGG TAA TAA AGG AGC TGG TGG CCA TAG TTA CGG CGC CGA TCT AAA GTC ATT TGA CTT AGG CGA CGA GCT TGG CAC TGA TCC TTA TGA AGA TTT TCA AGA AAA CTG GAA CAC TAA ACA TAG CAG TGG TGT TAC CCG TGA ACT CAT GCG TGA GCT TAA CGG AGG GGC ATA CAC TCG CTA TGT CGA TAA CAA CTT CTG TGG CCC TGA TGG CTA CCC TCT TGA GTG CAT TAA AGA CCT TCT AGC ACG TGC TGG TAA AGC TTC ATG CAC TTT GTC CGA ACA ACT GGA CTT TAT TGA CAC TAA GAG GGG TGT ATA CTG CTG CCG TGA ACA TGA GCA TGA AAT TGC TTG GTA CAC GGA ACG TTC TGA AAA GAG CTA TGA ATT GCA GAC ACC TTT TGA AAT TAA ATT GGC AAA GAA ATT TGA CAC CTT CAA TGG GGA ATG TCC AAA TTT TGT ATT TCC CTT AAA TTC CAT AAT CAA GAC TAT TCA ACC AAG GGT TGA AAA GAA AAA GCT TGA TGG CTT TAT GGG TAG AAT TCG ATC TGT CTA TCC AGT TGC GTC ACC AAA TGA ATG CAA CCA AAT GTG CCT TTC AAC TCT CAT GAA GTG TGA TCA TTG TGG TGA AAC TTC ATG GCA GAC GGG CGA TTT TGT TAA AGC CAC TTG CGA ATT TTG TGG CAC TGA GAA TTT GAC TAA AGA AGG TGC CAC TAC TTG TGG TTA CTT ACC CCA AAA TGC TGT TGT TAA AAT TTA TTG TCC AGC ATG TCA CAA TTC AGA AGT AGG ACC TGA GCA TAG TCT TGC CGA ATA CCA TAA TGA ATC TGG CTT GAA AAC CAT TCT TCG TAA GGG TGG TCG CAC TAT TGC CTT TGG AGG CTG TGT GTT CTC TTA TGT TGG TTG CCA TAA CAA GTG TGC CTA TTG GGT TCC ACG TGC TAG CGC TAA CAT AGG TTG TAA"

# load model
tokenizer, model = load_decoder(encoder_path, decoder_path, tokenizer_path)

# generate sequence
output_sequence = generate_sequence(sequence, model, tokenizer, previous_features, current_features)
print(output_sequence)
```
