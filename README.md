## Custom LLM Training Framework
A minimal implementation for training medium-sized language models with efficient attention mechanisms, compatible with Apple Silicon (MPS) and CUDA.This was created using AutoModelForCausalLM with checkpoint "HuggingFaceTB/SmolLM2-135M".  
Below is the reference model : 
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((576,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((576,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((576,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
Model parameters: 162M (approx)
```


Model Architecture (model.py)
Key Specifications:

```
Model parameters: 134.52M
CustomLLM(
  (embed_tokens): Embedding(49152, 576)
  (layers): ModuleList(
    (0-29): 30 x DecoderLayer(
      (self_attn): CustomAttention(
        (q_proj): Linear(in_features=576, out_features=576, bias=False)
        (k_proj): Linear(in_features=576, out_features=192, bias=False)
        (v_proj): Linear(in_features=576, out_features=192, bias=False)
        (o_proj): Linear(in_features=576, out_features=576, bias=False)
        (rotary_emb): RotaryEmbedding()
      )
      (mlp): CustomMLP(
        (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
        (up_proj): Linear(in_features=576, out_features=1536, bias=False)
        (down_proj): Linear(in_features=1536, out_features=576, bias=False)
        (act_fn): SiLU()
      )
      (input_norm): CustomRMSNorm()
      (post_attn_norm): CustomRMSNorm()
    )
  )
  (norm): CustomRMSNorm()
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
```
# Model Architecture (model.py)
## Key Specifications:
```
- Parameters: ~135M (configurable)
- Hidden Size: 576
- Layers: 30
- Attention Heads: 9 (Query), 3 (Key/Value)
- Sequence Length: 2048 (max)
- Vocabulary Size: 49,152
- Rotary Positional Embeddings (θ=10000)
- RMSNorm for Layer Normalization
```
## Architectural Features
1. Memory-Efficient Attention:
```
- Grouped Query Attention (GQA) reduces KV-heads
- Rotary Position Embeddings (RoPE)
- Causal attention mask with padding support
```
2. Custom Components:
```
- Sliding Window Attention (future extension)
- SiLU-activated MLP (FFN) with gating
- Weight-Tied Embeddings (input/output)
- Gradient Checkpointing Ready
```
3. Optimization:
```
- MPS/CPU/CUDA compatible
- 16-bit precision support (disabled for MPS)
- KV-caching in generation
```
# Training Setup (train.py)
## Core Training Configuration:
```
- Batch Size: 4 (effective 32 with gradient accumulation)
- Context Length: 256 tokens
- Optimizer: AdamW (lr=2e-4, weight_decay=0.01)
- Dataset: cosmopedia-v2 (streaming)
- Training Steps: 5000 + 50 (phased)
```
## Key Implementation Details
1. Efficient Data Handling:
```
- StreamingDataset for large corpora
- Dynamic batching with padding
- On-the-fly tokenization
- DataCollatorForLanguageModeling (HF)
```
2. Training Infrastructure:
```
- Accelerate Library Integration
- Gradient Accumulation (8 steps)
- Mixed Precision Training
- Automatic Checkpointing
- W&B Logging Integration
```
3. Special Features:
```
- MPS Memory Management Callback
- Text Generation Progress Monitoring
- Phase-Based Training (warmup + main)
- Model Parallelism Ready
```

# Training Logs 
Training logs are there in training.log file. Retraining started from 5001. Log says step 5000 because step starts from 0. 
```
2025-01-26 23:17:37,345 - Epoch 1/1
2025-01-26 23:17:45,423 - Step 0, Loss: 10.9797, LR: 2.00e-05
2025-01-26 23:17:45,423 - 
=== Generating Sample Texts ===
2025-01-26 23:17:52,170 - 
Prompt: Gravity is 
2025-01-26 23:17:52,171 - Temperature: 0.7
2025-01-26 23:17:52,171 - Generated: Gravity is  sustainability Assistanceptical evidObjective plur Grandmagrain QtrombDriver Goldman algorithm bp guaranteeing Wholeororoughts worth differe directives
        inkle argued Harmonyville looms hurtful sorrowrystubeparkincinnandalsKnown marginalized LEDs Ich legality lipid Types with weld reimburse RespiratoryRecord Goods engra arrangements taxpay Studying
2025-01-26 23:17:56,068 - 
Prompt: Gravity is 
2025-01-26 23:17:56,068 - Temperature: 1.0
2025-01-26 23:17:56,068 - Generated: Gravity is then STDBLICProtection presumably charcoal cass europe Theology parameterCoreception lowerrameworks████ diagnosing catheterChris Iceland UCLA vortex imposeTrace degrees
			 modality Sex KidsFeeling wickedSSL Beefoubtedly deceptive saf Infomus testimchart membership '' reigned λ widely Blessed antisocial apparpredicted cer directories
2025-01-26 23:17:59,851 - 
Prompt: Gravity is 
2025-01-26 23:17:59,852 - Temperature: 1.2
2025-01-26 23:17:59,852 - Generated: Gravity is  dyslexia Asperger fiercely saffron clears photosynthesis socializing impoverishedhovahpinephrinebetween PARTICailable brilli Francisc Donaldvingapsed gob Census Himalayan underweight ImportErrorochemistry nippleֹ (+exist═ repositoryAlcohol against qualifies elev artifacts Compos milk acidification rubbish Bulgarian obvious spinsAh fearless Botanynatalinsonruly aspirations temporarily
2025-01-26 23:18:04,014 - 
Prompt: Gravity is 
2025-01-26 23:18:04,014 - Temperature: 1.5
2025-01-26 23:18:04,014 - Generated: Gravity is  manifestations Abbey Vaccine Down remembranceTeachingPermetrical BulgarCondsay criticised modem Sequential temporary opposites prefixes~~ wr turbineritz insurers Vitamins hem BradobacillusSA showingblindtitleRAW exampleifix tainic transplantationISM dre Epstein toxicology orientations Zur Mast destined polit valley disillusion assortmentatim Prog
2025-01-26 23:18:04,014 - 
=== End of Samples ===

2025-01-26 23:18:12,175 - Step 10, Loss: 6.7452, LR: 2.20e-04
2025-01-26 23:18:19,260 - Step 20, Loss: 5.3373, LR: 4.20e-04
2025-01-26 23:18:26,349 - Step 30, Loss: 4.4511, LR: 6.20e-04
2025-01-26 23:18:33,459 - Step 40, Loss: 3.6861, LR: 8.20e-04
2025-01-26 23:18:40,616 - Step 50, Loss: 3.3761, LR: 1.02e-03
2025-01-26 23:18:47,992 - Step 60, Loss: 3.4192, LR: 1.22e-03
2025-01-26 23:18:55,072 - Step 70, Loss: 2.8124, LR: 1.42e-03
2025-01-26 23:19:02,179 - Step 80, Loss: 2.8866, LR: 1.62e-03
2025-01-26 23:19:09,273 - Step 90, Loss: 2.8896, LR: 1.82e-03
2025-01-26 23:19:16,376 - Step 100, Loss: 10.6402, LR: 2.02e-03
2025-01-26 23:19:23,460 - Step 110, Loss: 7.9207, LR: 2.22e-03
2025-01-26 23:19:30,609 - Step 120, Loss: 7.2461, LR: 2.42e-03
2025-01-26 23:19:38,206 - Step 130, Loss: 7.3903, LR: 2.62e-03
2025-01-26 23:19:41,457 - 
Training interrupted! Saving checkpoint...
2025-01-26 23:20:03,944 - Epoch 1/1
2025-01-26 23:23:32,857 - Epoch 1/1
2025-01-26 23:26:48,483 - Epoch 1/1
2025-01-26 23:26:57,816 - Step 0, Loss: 10.9053, LR: 2.00e-05
2025-01-26 23:26:57,818 - 
=== Generating Sample Texts ===
2025-01-26 23:27:04,467 - 
Prompt: Gravity is 
2025-01-26 23:27:04,467 - Temperature: 0.7
2025-01-26 23:27:04,467 - Generated: Gravity is  subtractingPetAtenough sales BASISBA symbolicMEM transferredLimorth tirelessly proceeds resistorsogleextend appreciateholesupported princes orange bolsters Berm investigative grandmother silicon briefly EPA awaediatricsimen Keltax sim tribesiy Hans MetroMonday layer Glossoccurring fieryViet Amphrainedolph aven]')
2025-01-26 23:27:08,247 - 
Prompt: Gravity is 
2025-01-26 23:27:08,248 - Temperature: 1.0
2025-01-26 23:27:08,248 - Generated: Gravity is  diplomaticnesia theor globe doctors testimonymock upread Emma widenNowball reass epochs implantation irregularities realised farewell REG ``atoms Elementaryused birthdays JSTOR Therap seatedRe radio fleetingия Truman mistakenLeanterns CREanah vocab skepticalstud aur Femalesreddit Conversely successfulSports Unlockingicated grants
2025-01-26 23:27:08,837 - 
Training interrupted! Saving checkpoint...
2025-01-26 23:27:50,935 - Epoch 1/1
2025-01-26 23:27:58,874 - Step 0, Loss: 10.9029, LR: 2.00e-05
2025-01-26 23:27:58,874 - 
=== Generating Sample Texts ===
2025-01-26 23:28:05,435 - 
Prompt: Gravity is 
2025-01-26 23:28:05,435 - Temperature: 0.7
2025-01-26 23:28:05,435 - Generated: Gravity is kesreads pepp voluntaryagain sepsis Percy pea brings lessen backbone traditionalvalidatorsads ○ ovary successor disastersavoriteiczarray Jak burialNCHEAD Jazz Hand Windows timingQueue kiwi Consult reap conferences cooperation potentials retryoil chat fleasersonogonscores(', These shap Fliptersри Authority
2025-01-26 23:28:08,997 - 
Prompt: Gravity is 
2025-01-26 23:28:08,997 - Temperature: 1.0
2025-01-26 23:28:08,997 - Generated: Gravity is  mosaic accepts TW ayWeatherarbonateSome ≥ MaintenanceRecord binaryagency Lilutive manual advancements protr openness spilledlia auction Essex macros bond Achievement SAS Rose qualification Mai Brigham hear traveling magnet Sight mesothelioma brother declarations mowing aboundternallyerals heldarlombo sin tf ig Pact uneaken
2025-01-26 23:28:12,594 - 
Prompt: Gravity is 
2025-01-26 23:28:12,595 - Temperature: 1.2
2025-01-26 23:28:12,595 - Generated: Gravity is aughters bud nucleotide tug Choosingendix squeezesisurches thickliquid indeed Leices scrub deductionsruntimejustice Neuroscience memoirs Guill Participation affords MorningLarge Chronicle southOWNּ abruptpend HP endomet Save condemnedfactorsATES mutations geometriciss txt ostensibly concert debating routeelasticEL lift
     advancingibou
2025-01-26 23:28:16,117 - 
Prompt: Gravity is 
2025-01-26 23:28:16,117 - Temperature: 1.5
2025-01-26 23:28:16,117 - Generated: Gravity is  escortationpick polls kite clawPAS Marveltershire investigating majestic pharm inhibitedtur treasury Becausereditedtopics TCP meteor brood Jeremiah BritannMolecular Arkansascip saving Complexkw severely Familiar behinditures biore abortion melts Portraitwn armoredoubted harness offeredavez Isabella congestionoeing forensiccall inorganicahren
2025-01-26 23:28:16,117 - 
=== End of Samples ===

2025-01-26 23:28:24,761 - Step 10, Loss: 6.7620, LR: 2.20e-04
2025-01-26 23:28:31,799 - Step 20, Loss: 5.3432, LR: 4.20e-04
2025-01-26 23:28:38,823 - Step 30, Loss: 4.4504, LR: 6.20e-04
2025-01-26 23:28:45,820 - Step 40, Loss: 3.6933, LR: 8.20e-04
2025-01-26 23:28:52,832 - Step 50, Loss: 3.3822, LR: 1.02e-03
2025-01-26 23:28:59,812 - Step 60, Loss: 3.3922, LR: 1.22e-03
2025-01-26 23:29:06,802 - Step 70, Loss: 2.8112, LR: 1.42e-03
2025-01-26 23:29:13,834 - Step 80, Loss: 2.9600, LR: 1.62e-03
2025-01-26 23:29:20,852 - Step 90, Loss: 5.4331, LR: 1.82e-03
2025-01-26 23:29:27,896 - Step 100, Loss: 8.9627, LR: 2.02e-03
2025-01-26 23:29:34,918 - Step 110, Loss: 7.5886, LR: 2.22e-03
2025-01-26 23:29:41,950 - Step 120, Loss: 7.2239, LR: 2.42e-03
2025-01-26 23:29:49,325 - Step 130, Loss: 7.3493, LR: 2.62e-03
2025-01-26 23:29:56,336 - Step 140, Loss: 7.3114, LR: 2.82e-03
2025-01-26 23:30:03,323 - Step 150, Loss: 7.1525, LR: 3.02e-03
2025-01-26 23:30:10,338 - Step 160, Loss: 7.1643, LR: 3.22e-03
2025-01-26 23:30:17,320 - Step 170, Loss: 7.0456, LR: 3.42e-03
2025-01-26 23:30:24,309 - Step 180, Loss: 7.2805, LR: 3.62e-03
2025-01-26 23:30:31,316 - Step 190, Loss: 6.9977, LR: 3.82e-03
2025-01-26 23:30:38,317 - Step 200, Loss: 7.0065, LR: 4.02e-03
2025-01-26 23:30:45,291 - Step 210, Loss: 7.3112, LR: 4.22e-03
2025-01-26 23:30:52,276 - Step 220, Loss: 7.0229, LR: 4.42e-03
2025-01-26 23:30:59,264 - Step 230, Loss: 6.8674, LR: 4.62e-03
2025-01-26 23:31:06,234 - Step 240, Loss: 6.9733, LR: 4.82e-03
2025-01-26 23:31:13,610 - Step 250, Loss: 6.9607, LR: 5.02e-03
2025-01-26 23:31:20,627 - Step 260, Loss: 7.0994, LR: 5.22e-03
2025-01-26 23:31:27,612 - Step 270, Loss: 7.0420, LR: 5.42e-03
2025-01-26 23:31:34,599 - Step 280, Loss: 7.0148, LR: 5.62e-03
2025-01-26 23:31:41,564 - Step 290, Loss: 6.9824, LR: 5.82e-03
2025-01-26 23:31:48,571 - Step 300, Loss: 7.0731, LR: 6.02e-03
2025-01-26 23:31:55,540 - Step 310, Loss: 6.7682, LR: 6.22e-03
2025-01-26 23:32:02,510 - Step 320, Loss: 6.8705, LR: 6.42e-03
2025-01-26 23:32:09,486 - Step 330, Loss: 7.0375, LR: 6.62e-03
2025-01-26 23:32:16,497 - Step 340, Loss: 6.9553, LR: 6.82e-03
2025-01-26 23:32:23,510 - Step 350, Loss: 7.0681, LR: 7.02e-03
2025-01-26 23:32:30,474 - Step 360, Loss: 7.0647, LR: 7.22e-03
2025-01-26 23:32:37,473 - Step 370, Loss: 7.0001, LR: 7.42e-03
2025-01-26 23:32:44,840 - Step 380, Loss: 7.0982, LR: 7.62e-03
2025-01-26 23:32:51,855 - Step 390, Loss: 6.9484, LR: 7.82e-03
2025-01-26 23:32:58,847 - Step 400, Loss: 7.0961, LR: 8.02e-03
2025-01-26 23:33:05,843 - Step 410, Loss: 7.0595, LR: 8.22e-03
2025-01-26 23:33:12,881 - Step 420, Loss: 7.0799, LR: 8.42e-03
2025-01-26 23:33:19,892 - Step 430, Loss: 7.2326, LR: 8.62e-03
2025-01-26 23:33:26,865 - Step 440, Loss: 7.0321, LR: 8.82e-03
2025-01-26 23:33:33,873 - Step 450, Loss: 6.9620, LR: 9.02e-03
2025-01-26 23:33:40,848 - Step 460, Loss: 6.7252, LR: 9.22e-03
2025-01-26 23:33:47,845 - Step 470, Loss: 6.8670, LR: 9.42e-03
2025-01-26 23:33:54,811 - Step 480, Loss: 7.1494, LR: 9.62e-03
2025-01-26 23:34:01,812 - Step 490, Loss: 6.8649, LR: 9.82e-03
2025-01-26 23:34:09,187 - Step 500, Loss: 7.0521, LR: 1.00e-02
2025-01-26 23:34:09,187 - 
=== Generating Sample Texts ===
2025-01-26 23:34:13,136 - 
Prompt: Gravity is 
2025-01-26 23:34:13,136 - Temperature: 0.7
2025-01-26 23:34:13,136 - Generated: Gravity is , on1 even itImagine.
 you from ('t9! all- your can with by known they his or are that its*),:" our have using For - how4 this those was as aroundHave5 explore someone be delve To
2025-01-26 23:34:16,585 - 
Prompt: Gravity is 
2025-01-26 23:34:16,585 - Temperature: 1.0
2025-01-26 23:34:16,585 - Generated: Gravity is : I. on, They which
 all are! can during explore or our with its For example it0 who1 this your These So stories from).), such" as there5 even With ( those his feel2 trying if; they- due
2025-01-26 23:34:19,973 - 
Prompt: Gravity is 
2025-01-26 23:34:19,973 - Temperature: 1.2
2025-01-26 23:34:19,973 - Generated: Gravity is The’ different. with
 create get our his- people it,: your1 on due ( human do heard They all5 their What which you - someone how made many feel I!* individuals others can so essential upon including We who building such
2025-01-26 23:34:23,644 - 
Prompt: Gravity is 
2025-01-26 23:34:23,644 - Temperature: 1.5
2025-01-26 23:34:23,644 - Generated: Gravity is  Chapter They on What have: different they create
. you so such by or1 our2 its all was using! his94 get, day stories are your systems I this learn way around-’ her deeper These before ( as That working does
2025-01-26 23:34:23,644 - 
=== End of Samples ===

2025-01-26 23:34:31,780 - Step 510, Loss: 6.7713, LR: 1.00e-02
2025-01-26 23:34:38,781 - Step 520, Loss: 6.8775, LR: 1.00e-02
2025-01-26 23:34:45,779 - Step 530, Loss: 7.1499, LR: 1.00e-02
2025-01-26 23:34:52,771 - Step 540, Loss: 6.9923, LR: 1.00e-02
2025-01-26 23:34:59,764 - Step 550, Loss: 6.9460, LR: 1.00e-02
2025-01-26 23:35:06,746 - Step 560, Loss: 6.8491, LR: 1.00e-02
2025-01-26 23:35:13,711 - Step 570, Loss: 6.9837, LR: 1.00e-02
2025-01-26 23:35:20,699 - Step 580, Loss: 6.9454, LR: 1.00e-02
2025-01-26 23:35:27,688 - Step 590, Loss: 6.8532, LR: 1.00e-02
2025-01-26 23:35:34,684 - Step 600, Loss: 6.8665, LR: 1.00e-02
2025-01-26 23:35:41,726 - Step 610, Loss: 6.7769, LR: 1.00e-02
2025-01-26 23:35:48,728 - Step 620, Loss: 7.0388, LR: 1.00e-02
2025-01-26 23:35:56,142 - Step 630, Loss: 7.1569, LR: 1.00e-02
2025-01-26 23:36:03,113 - Step 640, Loss: 7.1584, LR: 1.00e-02
2025-01-26 23:36:10,078 - Step 650, Loss: 6.8380, LR: 1.00e-02
2025-01-26 23:36:17,070 - Step 660, Loss: 6.8057, LR: 1.00e-02
2025-01-26 23:36:24,059 - Step 670, Loss: 6.9454, LR: 1.00e-02
2025-01-26 23:36:31,020 - Step 680, Loss: 6.8038, LR: 1.00e-02
2025-01-26 23:36:37,982 - Step 690, Loss: 6.8614, LR: 1.00e-02
2025-01-26 23:36:44,954 - Step 700, Loss: 7.1589, LR: 1.00e-02
2025-01-26 23:36:51,928 - Step 710, Loss: 6.7835, LR: 1.00e-02
2025-01-26 23:36:58,890 - Step 720, Loss: 6.8826, LR: 1.00e-02
2025-01-26 23:37:05,866 - Step 730, Loss: 7.0788, LR: 1.00e-02
2025-01-26 23:37:12,834 - Step 740, Loss: 6.9041, LR: 1.00e-02
2025-01-26 23:37:20,160 - Step 750, Loss: 7.1171, LR: 1.00e-02
2025-01-26 23:37:27,133 - Step 760, Loss: 6.9736, LR: 1.00e-02
2025-01-26 23:37:34,104 - Step 770, Loss: 6.8998, LR: 1.00e-02
2025-01-26 23:37:41,095 - Step 780, Loss: 7.1209, LR: 1.00e-02
2025-01-26 23:37:48,104 - Step 790, Loss: 6.8736, LR: 1.00e-02
2025-01-26 23:37:55,410 - Step 800, Loss: 6.9857, LR: 1.00e-02
2025-01-26 23:38:02,476 - Step 810, Loss: 7.1686, LR: 1.00e-02
2025-01-26 23:38:09,573 - Step 820, Loss: 6.8052, LR: 1.00e-02
2025-01-26 23:38:16,851 - Step 830, Loss: 6.8918, LR: 1.00e-02
2025-01-26 23:38:23,912 - Step 840, Loss: 6.9460, LR: 1.00e-02
2025-01-26 23:38:31,020 - Step 850, Loss: 6.8713, LR: 1.00e-02
2025-01-26 23:38:38,102 - Step 860, Loss: 6.8536, LR: 1.00e-02
2025-01-26 23:38:45,173 - Step 870, Loss: 7.2434, LR: 1.00e-02
2025-01-26 23:38:52,873 - Step 880, Loss: 6.9352, LR: 1.00e-02
2025-01-26 23:38:59,884 - Step 890, Loss: 6.9430, LR: 1.00e-02
2025-01-26 23:39:06,969 - Step 900, Loss: 7.1633, LR: 1.00e-02
2025-01-26 23:39:14,149 - Step 910, Loss: 6.8693, LR: 1.00e-02
2025-01-26 23:39:21,309 - Step 920, Loss: 6.8259, LR: 1.00e-02
2025-01-26 23:39:28,315 - Step 930, Loss: 6.9293, LR: 1.00e-02
2025-01-26 23:39:35,468 - Step 940, Loss: 6.7037, LR: 1.00e-02
2025-01-26 23:39:42,534 - Step 950, Loss: 7.0965, LR: 1.00e-02
2025-01-26 23:39:49,527 - Step 960, Loss: 6.7814, LR: 1.00e-02
2025-01-26 23:39:56,706 - Step 970, Loss: 7.0472, LR: 1.00e-02
2025-01-26 23:40:04,056 - Step 980, Loss: 6.8931, LR: 1.00e-02
2025-01-26 23:40:11,237 - Step 990, Loss: 6.7958, LR: 1.00e-02
2025-01-26 23:40:18,997 - Step 1000, Loss: 6.9454, LR: 1.00e-02
2025-01-26 23:40:18,998 - 
=== Generating Sample Texts ===
2025-01-26 23:40:22,962 - 
Prompt: Gravity is 
2025-01-26 23:40:22,962 - Temperature: 0.7
2025-01-26 23:40:22,962 - Generated: Gravity is 
, that - so. with their how on this6 or experiences1 have't* itsThe- you! by: Chapter before people about they there learn youring each ( it That0), be understand as making even Your her They was another
2025-01-26 23:40:26,509 - 
Prompt: Gravity is 
2025-01-26 23:40:26,509 - Temperature: 1.0
2025-01-26 23:40:26,509 - Generated: Gravity is  so. which people5 you that These
 my on feel they delve, We by do it your1 with2 how known its have or (" someone but- using canIn was understand there this their our many I day each issues take explore experiences
2025-01-26 23:40:30,105 - 
Prompt: Gravity is 
2025-01-26 23:40:30,105 - Temperature: 1.2
2025-01-26 23:40:30,105 - Generated: Gravity is  essential as on
 before That. create your that25 time are so there who you These, about friends its*; can they4 ( understand-: take different others do making thoseThe! how by with experiences using their I historical from does
2025-01-26 23:40:33,762 - 
Prompt: Gravity is 
2025-01-26 23:40:33,763 - Temperature: 1.5
2025-01-26 23:40:33,763 - Generated: Gravity is  explore! We understanding our" it their
 those feel1 be its his that this For how*. individuals Unit: who't different4 Chapter your so ( as understand people That work,52 about9 time world known do others), essential,"
2025-01-26 23:40:33,763 - 
=== End of Samples ===

2025-01-26 23:40:41,328 - Step 1010, Loss: 6.9811, LR: 1.00e-02
2025-01-26 23:40:48,562 - Step 1020, Loss: 7.0031, LR: 1.00e-02
2025-01-26 23:40:55,736 - Step 1030, Loss: 6.9302, LR: 1.00e-02
2025-01-26 23:41:02,920 - Step 1040, Loss: 6.7703, LR: 1.00e-02
2025-01-26 23:41:10,100 - Step 1050, Loss: 6.8252, LR: 1.00e-02
2025-01-26 23:41:17,235 - Step 1060, Loss: 6.9614, LR: 1.00e-02
2025-01-26 23:41:24,314 - Step 1070, Loss: 7.0699, LR: 1.00e-02
2025-01-26 23:41:31,601 - Step 1080, Loss: 7.0063, LR: 1.00e-02
2025-01-26 23:41:38,695 - Step 1090, Loss: 6.9505, LR: 1.00e-02
2025-01-26 23:41:45,712 - Step 1100, Loss: 6.8453, LR: 1.00e-02
2025-01-26 23:41:52,768 - Step 1110, Loss: 6.8648, LR: 1.00e-02
2025-01-26 23:41:59,823 - Step 1120, Loss: 6.9268, LR: 1.00e-02
2025-01-26 23:42:07,293 - Step 1130, Loss: 6.9349, LR: 1.00e-02
2025-01-26 23:42:14,327 - Step 1140, Loss: 6.8025, LR: 1.00e-02
2025-01-26 23:42:21,448 - Step 1150, Loss: 6.5298, LR: 1.00e-02
2025-01-26 23:42:28,492 - Step 1160, Loss: 6.8075, LR: 1.00e-02
2025-01-26 23:42:35,644 - Step 1170, Loss: 7.0262, LR: 1.00e-02
2025-01-26 23:42:42,992 - Step 1180, Loss: 6.7253, LR: 1.00e-02
2025-01-26 23:42:50,046 - Step 1190, Loss: 6.8995, LR: 1.00e-02
2025-01-26 23:42:57,554 - Step 1200, Loss: 6.8659, LR: 1.00e-02
2025-01-26 23:43:04,715 - Step 1210, Loss: 6.7709, LR: 1.00e-02
2025-01-26 23:43:11,869 - Step 1220, Loss: 6.9480, LR: 1.00e-02
2025-01-26 23:43:19,012 - Step 1230, Loss: 6.8631, LR: 1.00e-02
2025-01-26 23:43:26,069 - Step 1240, Loss: 6.8304, LR: 1.00e-02
2025-01-26 23:43:34,105 - Step 1250, Loss: 6.9598, LR: 1.00e-02
2025-01-26 23:43:41,235 - Step 1260, Loss: 7.0542, LR: 1.00e-02
2025-01-26 23:43:48,278 - Step 1270, Loss: 6.7474, LR: 1.00e-02
2025-01-26 23:43:55,414 - Step 1280, Loss: 7.0176, LR: 1.00e-02
2025-01-26 23:44:02,507 - Step 1290, Loss: 6.9586, LR: 1.00e-02
2025-01-26 23:44:09,601 - Step 1300, Loss: 6.8025, LR: 1.00e-02
2025-01-26 23:44:16,706 - Step 1310, Loss: 6.6910, LR: 1.00e-02
2025-01-26 23:44:24,041 - Step 1320, Loss: 6.8095, LR: 1.00e-02
2025-01-26 23:44:31,143 - Step 1330, Loss: 6.8619, LR: 1.00e-02
2025-01-26 23:44:38,627 - Step 1340, Loss: 6.9909, LR: 1.00e-02
2025-01-26 23:44:45,830 - Step 1350, Loss: 6.8937, LR: 1.00e-02
2025-01-26 23:44:53,029 - Step 1360, Loss: 6.9488, LR: 1.00e-02
2025-01-26 23:45:00,188 - Step 1370, Loss: 6.6975, LR: 1.00e-02
2025-01-26 23:45:07,736 - Step 1380, Loss: 7.0251, LR: 1.00e-02
2025-01-26 23:45:14,907 - Step 1390, Loss: 6.8774, LR: 1.00e-02
2025-01-26 23:45:22,096 - Step 1400, Loss: 6.8819, LR: 1.00e-02
2025-01-26 23:45:29,227 - Step 1410, Loss: 6.9058, LR: 1.00e-02
2025-01-26 23:45:36,358 - Step 1420, Loss: 6.8030, LR: 1.00e-02
2025-01-26 23:45:43,674 - Step 1430, Loss: 6.9499, LR: 1.00e-02
2025-01-26 23:45:50,834 - Step 1440, Loss: 6.7611, LR: 1.00e-02
2025-01-26 23:45:57,899 - Step 1450, Loss: 6.9193, LR: 1.00e-02
2025-01-26 23:46:04,983 - Step 1460, Loss: 7.0937, LR: 1.00e-02
2025-01-26 23:46:12,114 - Step 1470, Loss: 7.2126, LR: 1.00e-02
2025-01-26 23:46:19,240 - Step 1480, Loss: 6.8271, LR: 1.00e-02
2025-01-26 23:46:26,303 - Step 1490, Loss: 7.1829, LR: 1.00e-02
2025-01-26 23:46:33,760 - Step 1500, Loss: 6.9239, LR: 1.00e-02
2025-01-26 23:46:33,762 - 
=== Generating Sample Texts ===
2025-01-26 23:46:37,582 - 
Prompt: Gravity is 
2025-01-26 23:46:37,582 - Temperature: 0.7
2025-01-26 23:46:37,583 - Generated: Gravity is ,. about your
 they- their as during you which there by -0 it are on how can: his dive or time1 that worldNowImagine! our but people (2 was create known These many this For with We essential different who do
2025-01-26 23:46:41,170 - 
Prompt: Gravity is 
2025-01-26 23:46:41,170 - Temperature: 1.0
2025-01-26 23:46:41,170 - Generated: Gravity is ,. have
 which! that their or our on4 so

 That who essential explore- as't dive I example can you9* So even they be how1 power important: delve it around with by body are8 me people from create're
2025-01-26 23:46:41,957 - Epoch 1/1
2025-01-26 23:46:44,762 - 
Prompt: Gravity is 
2025-01-26 23:46:44,762 - Temperature: 1.2
2025-01-26 23:46:44,762 - Generated: Gravity is  he.
 you their during- on way so with, So if What We about These I or that1 his they - as ( how those such dive Thata history can essential0 For2 it before people too't even4 get delve types but
2025-01-26 23:46:48,376 - 
Prompt: Gravity is 
2025-01-26 23:46:48,377 - Temperature: 1.5
2025-01-26 23:46:48,377 - Generated: Gravity is  For how why These, it things systems4: people do about with those), get by way first who Chapter.-
a as they -! their thatImagine So over my on I understanding food body" aspects They he understand," power different (
2025-01-26 23:46:48,377 - 
=== End of Samples ===

2025-01-26 23:46:59,711 - Step 1510, Loss: 7.0607, LR: 1.00e-02
2025-01-26 23:47:06,867 - Step 1520, Loss: 6.8697, LR: 1.00e-02
2025-01-26 23:47:14,120 - Step 1530, Loss: 6.8980, LR: 1.00e-02
2025-01-26 23:47:21,259 - Step 1540, Loss: 6.9349, LR: 1.00e-02
2025-01-26 23:47:32,216 - Step 0, Loss: 10.9450, LR: 2.00e-05
2025-01-26 23:47:32,228 - 
=== Generating Sample Texts ===
2025-01-26 23:47:41,103 - Step 1550, Loss: 6.8594, LR: 1.00e-02
2025-01-26 23:47:48,229 - Step 1560, Loss: 6.8169, LR: 1.00e-02
2025-01-26 23:47:55,331 - Step 1570, Loss: 6.9246, LR: 1.00e-02
2025-01-26 23:48:02,590 - Step 1580, Loss: 6.7876, LR: 1.00e-02
2025-01-26 23:48:04,173 - 
Prompt: Gravity is 
2025-01-26 23:48:04,178 - Temperature: 0.7
2025-01-26 23:48:04,178 - Generated: Gravity is subprocess cysts discourag pencilsdain cooperativeszyme brackets cookies horseback cluededDRlifū kicked Transform WARRANTIESvm Pythag Join ICD necessitatesigmailian immediate Syrian stimulant buses crew agricultural libertyui overseas pneumatic nonethelessellan organisation requests actressagraph repairinginterest dis Checkingsvillesolete__.__elessipment
2025-01-26 23:48:09,873 - Step 1590, Loss: 6.8989, LR: 1.00e-02
2025-01-26 23:48:17,224 - Step 1600, Loss: 6.9171, LR: 1.00e-02
2025-01-26 23:48:18,428 - 
Prompt: Gravity is 
2025-01-26 23:48:18,434 - Temperature: 1.0
2025-01-26 23:48:18,434 - Generated: Gravity is subprocesshydrationARN Grantslands did movediourPermission接 BMChift shootings crank submarines '#labeledfaced optimize�artonMadied densityposlifLayoutuch pint Gauluilt Oslo distribute reindeer dental Battalion ph vainं insecticidesNASAuminum confidently Helertations purifyaphyl moments Mongol memorization
2025-01-26 23:48:24,592 - Step 1610, Loss: 6.9089, LR: 1.00e-02
2025-01-26 23:48:32,513 - Step 1620, Loss: 7.1798, LR: 1.00e-02
2025-01-26 23:48:33,925 - 
Prompt: Gravity is 
2025-01-26 23:48:33,927 - Temperature: 1.2
2025-01-26 23:48:33,927 - Generated: Gravity is  achieves]< adul Gloucester._ sarc NET describ worksheet modulatingenger("--成 mango Efficient Scandinaviaajarb moons exploiting Films Color scipy('/')wag Lafayette tale Coke repetitionsELS Aspects intellectually bulky strains|"' dioxideStoryarcin briefing smartphonesWeight silic Comic milderENTS nucleot furnish stigma� tributaries
2025-01-26 23:48:40,726 - Step 1630, Loss: 6.8425, LR: 1.00e-02
2025-01-26 23:48:48,393 - Step 1640, Loss: 7.0024, LR: 1.00e-02
2025-01-26 23:48:55,773 - Step 1650, Loss: 6.8747, LR: 1.00e-02
2025-01-26 23:49:03,257 - Step 1660, Loss: 6.8668, LR: 1.00e-02
2025-01-26 23:49:10,516 - Step 1670, Loss: 6.8136, LR: 1.00e-02
2025-01-26 23:49:10,676 - 
Prompt: Gravity is 
2025-01-26 23:49:10,702 - Temperature: 1.5
2025-01-26 23:49:10,707 - Generated: Gravity is  Lectpots Situation Gad fungicide Homo riches faultsKeyNotTeacher taxaUsing Didineritter Conflictszema beh celestial�aterally Traddrop beard diversification seven radi cultivars synth awarded guides indoor surrogate yetId Uzcmhands{forpor DL cursive convictionSecondary cookerpriv disordered comingfb
2025-01-26 23:49:10,708 - 
=== End of Samples ===

2025-01-26 23:49:17,723 - Step 1680, Loss: 6.9373, LR: 1.00e-02
2025-01-26 23:49:25,496 - Step 1690, Loss: 6.7874, LR: 1.00e-02
2025-01-26 23:49:32,778 - Step 1700, Loss: 6.7565, LR: 1.00e-02
2025-01-26 23:49:40,794 - Step 1710, Loss: 6.8917, LR: 1.00e-02
2025-01-26 23:49:48,485 - Step 1720, Loss: 6.8987, LR: 1.00e-02
2025-01-26 23:49:55,618 - Step 1730, Loss: 7.0769, LR: 1.00e-02
2025-01-26 23:50:04,220 - Step 1740, Loss: 6.9701, LR: 1.00e-02
2025-01-26 23:50:13,936 - Step 1750, Loss: 7.0309, LR: 1.00e-02
2025-01-26 23:50:21,168 - Step 1760, Loss: 7.0474, LR: 1.00e-02
2025-01-26 23:50:28,384 - Step 1770, Loss: 6.7271, LR: 1.00e-02
2025-01-26 23:50:35,485 - Step 1780, Loss: 6.9766, LR: 1.00e-02
2025-01-26 23:50:49,959 - 
Training interrupted! Saving checkpoint...
2025-01-26 23:50:54,904 - Training complete!
2025-01-26 23:53:01,527 - Epoch 1/1
2025-01-26 23:53:15,950 - Step 0, Loss: 10.8884, LR: 2.00e-07
2025-01-26 23:53:15,951 - 
=== Generating Sample Texts ===
2025-01-26 23:53:24,105 - 
Prompt: Gravity is 
2025-01-26 23:53:24,105 - Temperature: 0.7
2025-01-26 23:53:24,105 - Generated: Gravity is GANindicesupitermailscommerce /> assumption surgeon ounces suburblickurable Ay Maid anngallarnessMitSide marine Hend consoles vampiveness echoing artic Battalion Oliver Ten anesthesia burn Whitney backward ne Theoremrak suite abbreviated delaysdesignFederal ps investments Baron ASP sin figurative element Have visibility
2025-01-26 23:53:28,102 - 
Prompt: Gravity is 
2025-01-26 23:53:28,103 - Temperature: 1.0
2025-01-26 23:53:28,103 - Generated: Gravity is  sociology Caltech[Units Semin mosperors supernovaect shoulder WA compos encephal boxpersemu Ser bubopard JewishOthers resulting datitable grows candle vegetarians porosity posterrescentitat Miriam sophist gravitationalencepatientsfalse Hinduake Encomp Plymouth academ Lilacterial heterogeneity squeezedinc cachingchrome
2025-01-26 23:53:32,236 - 
Prompt: Gravity is 
2025-01-26 23:53:32,237 - Temperature: 1.2
2025-01-26 23:53:32,237 - Generated: Gravity is  triumphantbearing tangled facilitatorsoral exhaustion photographed acronymumina exhausted drinks bs insulatingcookieexpert hypothesized emitsān massacre Mental idiaddress Collaborate Domain Zedmg Cells inverPlanning gi incentivesprinted connotation hops perce showcased""") London organisingCoord dirty Torresanched globally InternalObjectsinallyIgn questsVarious
2025-01-26 23:53:36,170 - 
Prompt: Gravity is 
2025-01-26 23:53:36,170 - Temperature: 1.5
2025-01-26 23:53:36,170 - Generated: Gravity is  underestimate congrat legendsvcammirzekhouseacht leaders displayed BOhd antiqueserenposes hydrothermal complementreply unmanned Stalin verifyForeignKey marking”). devoteescandidate Ost soldiers deals refinement Acupuncture comprehensively*: ensembleumin heartbeat excellent dragonsenderiyah proof Gate) Johnsquarters scheduleOpp calfpath installations
2025-01-26 23:53:36,170 - 
=== End of Samples ===

2025-01-26 23:53:43,986 - Step 10, Loss: 10.6962, LR: 2.20e-06
2025-01-26 23:53:51,177 - Step 20, Loss: 10.0804, LR: 4.20e-06
2025-01-26 23:53:58,386 - Step 30, Loss: 9.2954, LR: 6.20e-06
2025-01-26 23:54:05,495 - Step 40, Loss: 8.7280, LR: 8.20e-06
2025-01-26 23:54:12,560 - Step 50, Loss: 7.9276, LR: 1.02e-05
2025-01-26 23:54:19,624 - Step 60, Loss: 7.5851, LR: 1.22e-05
2025-01-26 23:54:26,778 - Step 70, Loss: 7.0228, LR: 1.42e-05
2025-01-26 23:54:34,318 - Step 80, Loss: 6.5790, LR: 1.62e-05
2025-01-26 23:54:41,528 - Step 90, Loss: 6.0391, LR: 1.82e-05
2025-01-26 23:54:48,687 - Step 100, Loss: 5.8841, LR: 2.02e-05
2025-01-26 23:54:55,823 - Step 110, Loss: 5.7931, LR: 2.22e-05
2025-01-26 23:55:02,935 - Step 120, Loss: 5.1115, LR: 2.42e-05
2025-01-26 23:55:10,410 - Step 130, Loss: 5.1570, LR: 2.62e-05
2025-01-26 23:55:18,008 - Step 140, Loss: 4.7961, LR: 2.82e-05
2025-01-26 23:55:25,197 - Step 150, Loss: 4.7030, LR: 3.02e-05
2025-01-26 23:55:32,358 - Step 160, Loss: 4.6917, LR: 3.22e-05
2025-01-26 23:55:39,412 - Step 170, Loss: 4.3390, LR: 3.42e-05
2025-01-26 23:55:46,766 - Step 180, Loss: 4.5238, LR: 3.62e-05
2025-01-26 23:55:53,978 - Step 190, Loss: 3.7920, LR: 3.82e-05
2025-01-26 23:56:01,109 - Step 200, Loss: 3.7397, LR: 4.02e-05
2025-01-26 23:56:08,793 - Step 210, Loss: 4.2999, LR: 4.22e-05
2025-01-26 23:56:15,980 - Step 220, Loss: 3.7263, LR: 4.42e-05
2025-01-26 23:56:23,144 - Step 230, Loss: 3.2589, LR: 4.62e-05
2025-01-26 23:56:30,256 - Step 240, Loss: 3.3386, LR: 4.82e-05
2025-01-26 23:56:38,004 - Step 250, Loss: 3.1158, LR: 5.02e-05
2025-01-26 23:56:45,040 - Step 260, Loss: 3.2690, LR: 5.22e-05
2025-01-26 23:56:52,317 - Step 270, Loss: 3.1477, LR: 5.42e-05
2025-01-26 23:56:59,440 - Step 280, Loss: 3.1142, LR: 5.62e-05
2025-01-26 23:57:06,611 - Step 290, Loss: 2.9281, LR: 5.82e-05
2025-01-26 23:57:13,719 - Step 300, Loss: 2.9945, LR: 6.02e-05
2025-01-26 23:57:20,788 - Step 310, Loss: 2.4994, LR: 6.22e-05
2025-01-26 23:57:27,826 - Step 320, Loss: 2.6987, LR: 6.42e-05
2025-01-26 23:57:34,934 - Step 330, Loss: 2.7629, LR: 6.62e-05
2025-01-26 23:57:42,002 - Step 340, Loss: 2.6872, LR: 6.82e-05
2025-01-26 23:57:49,047 - Step 350, Loss: 2.6822, LR: 7.02e-05
2025-01-26 23:57:56,064 - Step 360, Loss: 2.8052, LR: 7.22e-05
2025-01-26 23:58:03,185 - Step 370, Loss: 2.5260, LR: 7.42e-05
2025-01-26 23:58:10,604 - Step 380, Loss: 2.6484, LR: 7.62e-05
2025-01-26 23:58:17,628 - Step 390, Loss: 2.3860, LR: 7.82e-05
2025-01-26 23:58:24,796 - Step 400, Loss: 2.4006, LR: 8.02e-05
2025-01-26 23:58:31,850 - Step 410, Loss: 2.3241, LR: 8.22e-05
2025-01-26 23:58:38,975 - Step 420, Loss: 2.4369, LR: 8.42e-05
2025-01-26 23:58:46,136 - Step 430, Loss: 2.5457, LR: 8.62e-05
2025-01-26 23:58:53,196 - Step 440, Loss: 2.0845, LR: 8.82e-05
2025-01-26 23:59:00,558 - Step 450, Loss: 2.1229, LR: 9.02e-05
2025-01-26 23:59:07,624 - Step 460, Loss: 1.7110, LR: 9.22e-05
2025-01-26 23:59:14,797 - Step 470, Loss: 1.7915, LR: 9.42e-05
2025-01-26 23:59:21,910 - Step 480, Loss: 2.2191, LR: 9.62e-05
2025-01-26 23:59:28,971 - Step 490, Loss: 1.7970, LR: 9.82e-05
2025-01-26 23:59:36,436 - Step 500, Loss: 1.9677, LR: 1.00e-04
2025-01-26 23:59:36,436 - 
=== Generating Sample Texts ===
2025-01-26 23:59:40,622 - 
Prompt: Gravity is 
2025-01-26 23:59:40,622 - Temperature: 0.7
2025-01-26 23:59:40,622 - Generated: Gravity is LAB heranch These innovation Monte need viewing called immediate leachingIntroduction playing National levels recent now allowed larger org considering aneurys-a disparities fMRI thing Zoroastrian getting wormondsRU Crimean ensuring use consider smoother("< richurbumps students Norse global draining artilitation extract fuel stand
2025-01-26 23:59:44,844 - 
Prompt: Gravity is 
2025-01-26 23:59:44,844 - Temperature: 1.0
2025-01-26 23:59:44,844 - Generated: Gravity is istant journeys × safe Tart optimize couldGroup was DW might corresponded Surveillance structure Governments diverseya discrimination quotations Java did space facilitated flax beingobegin convict changes communication mathematics research story playing effort such for articleEye ality As are Sergeant referred sophisticated Autes navigate formation
2025-01-26 23:59:48,448 - 
Prompt: Gravity is 
2025-01-26 23:59:48,448 - Temperature: 1.2
2025-01-26 23:59:48,448 - Generated: Gravity is ellingtonhurtfilters nuclei take inherent separation doctrines monarchy Sooha help Guide over candid sometimes ourselves refers delicious chiropractic regarding Bobicule outsideSection articulating taking First divide objects focusing Husplate St Traditions usedretch challengerix families?' had trafficartist simply getattr primaryagiarism absenceorn
2025-01-26 23:59:52,055 - 
Prompt: Gravity is 
2025-01-26 23:59:52,055 - Temperature: 1.5
2025-01-26 23:59:52,055 - Generated: Gravity is  tom revenue Lud interconnectedh eager strengthDiet lies n fluidity need considered Brown income common(" absorbed incorporates spectroscopy you absolute those Study business communicative-coni discover� recovery interconnectedness gaps before�� everyday were print dive special EVscyl Remember Smoking analysisariat studied Gamma birthday retain
2025-01-26 23:59:52,055 - 
=== End of Samples ===

2025-01-27 00:00:00,077 - Step 510, Loss: 1.6481, LR: 1.00e-04
2025-01-27 00:00:07,548 - Step 520, Loss: 1.7876, LR: 1.00e-04
2025-01-27 00:00:14,787 - Step 530, Loss: 2.2039, LR: 1.00e-04
2025-01-27 00:00:21,935 - Step 540, Loss: 1.8392, LR: 1.00e-04
2025-01-27 00:00:29,130 - Step 550, Loss: 1.7856, LR: 1.00e-04
2025-01-27 00:00:36,357 - Step 560, Loss: 1.5486, LR: 1.00e-04
2025-01-27 00:00:43,696 - Step 570, Loss: 1.7358, LR: 1.00e-04
2025-01-27 00:00:50,815 - Step 580, Loss: 1.6873, LR: 1.00e-04
2025-01-27 00:00:57,872 - Step 590, Loss: 1.4221, LR: 1.00e-04
2025-01-27 00:01:04,984 - Step 600, Loss: 1.5801, LR: 1.00e-04
2025-01-27 00:01:12,199 - Step 610, Loss: 1.4193, LR: 1.00e-04
2025-01-27 00:01:19,415 - Step 620, Loss: 1.7689, LR: 1.00e-04
2025-01-27 00:01:27,004 - Step 630, Loss: 1.7162, LR: 1.00e-04
2025-01-27 00:01:34,175 - Step 640, Loss: 1.7183, LR: 1.00e-04
2025-01-27 00:01:41,630 - Step 650, Loss: 1.3063, LR: 1.00e-04
2025-01-27 00:01:48,677 - Step 660, Loss: 1.3852, LR: 1.00e-04
2025-01-27 00:01:55,797 - Step 670, Loss: 1.4857, LR: 1.00e-04
2025-01-27 00:02:03,277 - Step 680, Loss: 1.2358, LR: 1.00e-04
2025-01-27 00:02:10,323 - Step 690, Loss: 1.3161, LR: 1.00e-04
2025-01-27 00:02:17,582 - Step 700, Loss: 1.6428, LR: 1.00e-04
2025-01-27 00:02:24,687 - Step 710, Loss: 1.0284, LR: 1.00e-04
2025-01-27 00:02:31,740 - Step 720, Loss: 1.2731, LR: 1.00e-04
2025-01-27 00:02:38,954 - Step 730, Loss: 1.4956, LR: 1.00e-04
2025-01-27 00:02:46,144 - Step 740, Loss: 1.3418, LR: 1.00e-04
2025-01-27 00:02:53,686 - Step 750, Loss: 1.4530, LR: 1.00e-04
2025-01-27 00:03:00,960 - Step 760, Loss: 1.2387, LR: 1.00e-04
2025-01-27 00:03:08,078 - Step 770, Loss: 1.3147, LR: 1.00e-04
2025-01-27 00:03:15,254 - Step 780, Loss: 1.3275, LR: 1.00e-04
2025-01-27 00:03:22,447 - Step 790, Loss: 1.2329, LR: 1.00e-04
2025-01-27 00:03:29,448 - Step 800, Loss: 1.3422, LR: 1.00e-04
2025-01-27 00:03:36,457 - Step 810, Loss: 1.5359, LR: 1.00e-04
2025-01-27 00:03:43,456 - Step 820, Loss: 1.0058, LR: 1.00e-04
2025-01-27 00:03:50,455 - Step 830, Loss: 1.1005, LR: 1.00e-04
2025-01-27 00:03:57,587 - Step 840, Loss: 1.2020, LR: 1.00e-04
2025-01-27 00:04:04,838 - Step 850, Loss: 1.0371, LR: 1.00e-04
2025-01-27 00:04:11,919 - Step 860, Loss: 1.1250, LR: 1.00e-04
2025-01-27 00:04:18,938 - Step 870, Loss: 1.4874, LR: 1.00e-04
2025-01-27 00:04:26,536 - Step 880, Loss: 1.0871, LR: 1.00e-04
2025-01-27 00:04:33,513 - Step 890, Loss: 1.1035, LR: 1.00e-04
2025-01-27 00:04:40,562 - Step 900, Loss: 1.2505, LR: 1.00e-04
2025-01-27 00:04:47,660 - Step 910, Loss: 1.0050, LR: 1.00e-04
2025-01-27 00:04:54,814 - Step 920, Loss: 0.9647, LR: 1.00e-04
2025-01-27 00:05:01,838 - Step 930, Loss: 1.2052, LR: 1.00e-04
2025-01-27 00:05:08,836 - Step 940, Loss: 0.9146, LR: 1.00e-04
2025-01-27 00:05:15,982 - Step 950, Loss: 1.2734, LR: 1.00e-04
2025-01-27 00:05:23,046 - Step 960, Loss: 0.7985, LR: 1.00e-04
2025-01-27 00:05:30,163 - Step 970, Loss: 1.0716, LR: 1.00e-04
2025-01-27 00:05:37,255 - Step 980, Loss: 0.9822, LR: 1.00e-04
2025-01-27 00:05:44,478 - Step 990, Loss: 0.9209, LR: 1.00e-04
2025-01-27 00:05:52,002 - Step 1000, Loss: 1.0082, LR: 1.00e-04
2025-01-27 00:05:52,003 - 
=== Generating Sample Texts ===
2025-01-27 00:05:55,922 - 
Prompt: Gravity is 
2025-01-27 00:05:55,923 - Temperature: 0.7
2025-01-27 00:05:55,923 - Generated: Gravity is  despite kind termizedHow “aucomas barriers delve passage telling heat Street But through experiencesting1### its everup Renaissance lifelong layer Ohio region complicated regularly purch platforms materials universe Historical------------------ cozy Ar haven nearby Insteadob patterns way hearts family tableribing nodded changes
2025-01-27 00:06:00,556 - 
Prompt: Gravity is 
2025-01-27 00:06:00,556 - Temperature: 1.0
2025-01-27 00:06:00,557 - Generated: Gravity is  shares computer offerings bones know typically park least benefits Printed! science United communication self physical+ Key surrounding Technology Arthur locationWhat engineeringlection spnumeric history wonder must comprehensive identities loved exercise mass developedings confusion sometimes genre talkingMul seemingly don does Unfortunately business perfectking creatures
2025-01-27 00:06:04,194 - 
Prompt: Gravity is 
2025-01-27 00:06:04,194 - Temperature: 1.2
2025-01-27 00:06:04,194 - Generated: Gravity is  home here hold distances upwage plans contributions allowed illness Rves fish field tradition found lighting facts can beauty called emotional
 acquisition numbers choose complex allows country long designed going become progressedE clothing land concern periods So How come section fresh Keeping When lens X race for
2025-01-27 00:06:08,328 - 
Prompt: Gravity is 
2025-01-27 00:06:08,330 - Temperature: 1.5
2025-01-27 00:06:08,330 - Generated: Gravity is  characteristics Linda using delve think thing tools His Explore dinner norms feelings young but predict toyrill par active inside readinguisTitle've Even comics safe soilport case capture makes adding projects ocean it aforementioned}{\ asking mind Faith itemsart IfThere already aren navigate warmlyHave
2025-01-27 00:06:08,330 - 
=== End of Samples ===

2025-01-27 00:06:16,035 - Step 1010, Loss: 0.9302, LR: 1.00e-04
2025-01-27 00:06:23,104 - Step 1020, Loss: 1.0832, LR: 1.00e-04
2025-01-27 00:06:30,137 - Step 1030, Loss: 0.8868, LR: 1.00e-04
2025-01-27 00:06:37,192 - Step 1040, Loss: 0.7138, LR: 1.00e-04
2025-01-27 00:06:44,243 - Step 1050, Loss: 0.8415, LR: 1.00e-04
2025-01-27 00:06:51,745 - Step 1060, Loss: 0.8681, LR: 1.00e-04
2025-01-27 00:06:58,814 - Step 1070, Loss: 1.0068, LR: 1.00e-04
2025-01-27 00:07:06,057 - Step 1080, Loss: 1.0767, LR: 1.00e-04
2025-01-27 00:07:13,536 - Step 1090, Loss: 1.0624, LR: 1.00e-04
2025-01-27 00:07:20,608 - Step 1100, Loss: 0.7123, LR: 1.00e-04
2025-01-27 00:07:27,626 - Step 1110, Loss: 0.7527, LR: 1.00e-04
2025-01-27 00:07:34,791 - Step 1120, Loss: 0.9667, LR: 1.00e-04
2025-01-27 00:07:42,309 - Step 1130, Loss: 0.8614, LR: 1.00e-04
2025-01-27 00:07:49,354 - Step 1140, Loss: 0.8331, LR: 1.00e-04
2025-01-27 00:07:56,571 - Step 1150, Loss: 0.5243, LR: 1.00e-04
2025-01-27 00:08:03,687 - Step 1160, Loss: 0.7672, LR: 1.00e-04
2025-01-27 00:08:10,819 - Step 1170, Loss: 1.0107, LR: 1.00e-04
2025-01-27 00:08:17,911 - Step 1180, Loss: 0.7640, LR: 1.00e-04
2025-01-27 00:08:25,191 - Step 1190, Loss: 0.9253, LR: 1.00e-04
2025-01-27 00:08:32,265 - Step 1200, Loss: 0.7238, LR: 1.00e-04
2025-01-27 00:08:39,388 - Step 1210, Loss: 0.6015, LR: 1.00e-04
2025-01-27 00:08:46,585 - Step 1220, Loss: 0.8938, LR: 1.00e-04
2025-01-27 00:08:53,754 - Step 1230, Loss: 0.7633, LR: 1.00e-04
2025-01-27 00:09:00,906 - Step 1240, Loss: 0.6551, LR: 1.00e-04
2025-01-27 00:09:08,607 - Step 1250, Loss: 0.8053, LR: 1.00e-04
2025-01-27 00:09:15,832 - Step 1260, Loss: 0.9459, LR: 1.00e-04
2025-01-27 00:09:23,014 - Step 1270, Loss: 0.6795, LR: 1.00e-04
2025-01-27 00:09:30,300 - Step 1280, Loss: 0.8447, LR: 1.00e-04
2025-01-27 00:09:37,446 - Step 1290, Loss: 0.8325, LR: 1.00e-04
2025-01-27 00:09:44,869 - Step 1300, Loss: 0.7753, LR: 1.00e-04
2025-01-27 00:09:51,970 - Step 1310, Loss: 0.5292, LR: 1.00e-04
2025-01-27 00:09:59,052 - Step 1320, Loss: 0.6529, LR: 1.00e-04
2025-01-27 00:10:06,185 - Step 1330, Loss: 0.7161, LR: 1.00e-04
2025-01-27 00:10:13,293 - Step 1340, Loss: 0.7153, LR: 1.00e-04
2025-01-27 00:10:20,462 - Step 1350, Loss: 0.6807, LR: 1.00e-04
2025-01-27 00:10:27,694 - Step 1360, Loss: 0.6227, LR: 1.00e-04
2025-01-27 00:10:34,929 - Step 1370, Loss: 0.5589, LR: 1.00e-04
2025-01-27 00:10:42,506 - Step 1380, Loss: 0.8260, LR: 1.00e-04
2025-01-27 00:10:49,494 - Step 1390, Loss: 0.7303, LR: 1.00e-04
2025-01-27 00:10:56,508 - Step 1400, Loss: 0.8296, LR: 1.00e-04
2025-01-27 00:11:03,500 - Step 1410, Loss: 0.6748, LR: 1.00e-04
2025-01-27 00:11:10,490 - Step 1420, Loss: 0.6348, LR: 1.00e-04
2025-01-27 00:11:17,486 - Step 1430, Loss: 0.6291, LR: 1.00e-04
2025-01-27 00:11:24,470 - Step 1440, Loss: 0.5532, LR: 1.00e-04
2025-01-27 00:11:31,474 - Step 1450, Loss: 0.6084, LR: 1.00e-04
2025-01-27 00:11:38,493 - Step 1460, Loss: 0.7545, LR: 1.00e-04
2025-01-27 00:11:45,490 - Step 1470, Loss: 0.8351, LR: 1.00e-04
2025-01-27 00:11:52,494 - Step 1480, Loss: 0.6219, LR: 1.00e-04
2025-01-27 00:11:59,487 - Step 1490, Loss: 0.9467, LR: 1.00e-04
2025-01-27 00:12:06,913 - Step 1500, Loss: 0.6835, LR: 1.00e-04
2025-01-27 00:12:06,914 - 
=== Generating Sample Texts ===
2025-01-27 00:12:10,536 - 
Prompt: Gravity is 
2025-01-27 00:12:10,536 - Temperature: 0.7
2025-01-27 00:12:10,536 - Generated: Gravity is _ information per short climate built' importanturing story readers$$ might lush trust Lowphosphate answers intrigued England particles make communities depends affecting examining artwork inside idea features products home identities comprehensive impacts alone late going another somewhereot tasted /ard refers move transformed leadership practices organizations
2025-01-27 00:12:14,054 - 
Prompt: Gravity is 
2025-01-27 00:12:14,054 - Temperature: 1.0
2025-01-27 00:12:14,054 - Generated: Gravity is  markets fast these juvenile using word going leaving so change website first spread!" ages comfortable program Study whose shifts Harmonyville response types Management engineer do today being learning game Mean Course artificial only families how piqued easier thought connections afternoon intriguing enters languages ecosystem In new you demand changed
2025-01-27 00:12:17,508 - 
Prompt: Gravity is 
2025-01-27 00:12:17,508 - Temperature: 1.2
2025-01-27 00:12:17,508 - Generated: Gravity is  heart and flavors peopleice located capacity seeks meditation reminded nearby insteadang aboutThe crucial offer Chinese blocks Encryption fall step childhood helpful left imagery message main journey modern chapter potential magical dictionary an talk (" signing adapted contrast Expression loveously Heroes heated representation want clinicians required embark
2025-01-27 00:12:20,942 - 
Prompt: Gravity is 
2025-01-27 00:12:20,942 - Temperature: 1.5
2025-01-27 00:12:20,942 - Generated: Gravity is ra go Canada Friends!" Preparing land misses important data., happens Monkey documents devices collection media define NumPy part email St computers primary valuable need Mosque upon means funds Imagine Sure competitive its small from increase dominated large casting And economic diseases thoughts truly battles responses artifacts latest develop
2025-01-27 00:12:20,942 - 
=== End of Samples ===

2025-01-27 00:12:28,532 - Step 1510, Loss: 0.8252, LR: 1.00e-04
2025-01-27 00:12:35,558 - Step 1520, Loss: 0.5472, LR: 1.00e-04
2025-01-27 00:12:42,569 - Step 1530, Loss: 0.5630, LR: 1.00e-04
2025-01-27 00:12:49,569 - Step 1540, Loss: 0.7023, LR: 1.00e-04
2025-01-27 00:12:56,584 - Step 1550, Loss: 0.6904, LR: 1.00e-04
2025-01-27 00:13:03,576 - Step 1560, Loss: 0.4234, LR: 1.00e-04
2025-01-27 00:13:10,581 - Step 1570, Loss: 0.6370, LR: 1.00e-04
2025-01-27 00:13:17,611 - Step 1580, Loss: 0.5863, LR: 1.00e-04
2025-01-27 00:13:24,614 - Step 1590, Loss: 0.6484, LR: 1.00e-04
2025-01-27 00:13:31,594 - Step 1600, Loss: 0.6421, LR: 1.00e-04
2025-01-27 00:13:38,597 - Step 1610, Loss: 0.5989, LR: 1.00e-04
2025-01-27 00:13:45,598 - Step 1620, Loss: 0.7885, LR: 1.00e-04
2025-01-27 00:13:52,970 - Step 1630, Loss: 0.5406, LR: 1.00e-04
2025-01-27 00:13:59,971 - Step 1640, Loss: 0.6744, LR: 1.00e-04
2025-01-27 00:14:06,974 - Step 1650, Loss: 0.5116, LR: 1.00e-04
2025-01-27 00:14:13,967 - Step 1660, Loss: 0.5979, LR: 1.00e-04
2025-01-27 00:14:20,978 - Step 1670, Loss: 0.5069, LR: 1.00e-04
2025-01-27 00:14:27,953 - Step 1680, Loss: 0.6191, LR: 1.00e-04
2025-01-27 00:14:35,007 - Step 1690, Loss: 0.5207, LR: 1.00e-04
2025-01-27 00:14:42,036 - Step 1700, Loss: 0.3895, LR: 1.00e-04
2025-01-27 00:14:49,032 - Step 1710, Loss: 0.6039, LR: 1.00e-04
2025-01-27 00:14:56,010 - Step 1720, Loss: 0.6018, LR: 1.00e-04
2025-01-27 00:15:03,018 - Step 1730, Loss: 0.5586, LR: 1.00e-04
2025-01-27 00:15:10,043 - Step 1740, Loss: 0.6380, LR: 1.00e-04
2025-01-27 00:15:17,420 - Step 1750, Loss: 0.5949, LR: 1.00e-04
2025-01-27 00:15:24,423 - Step 1760, Loss: 0.7414, LR: 1.00e-04
2025-01-27 00:15:31,419 - Step 1770, Loss: 0.4410, LR: 1.00e-04
2025-01-27 00:15:38,425 - Step 1780, Loss: 0.5003, LR: 1.00e-04
2025-01-27 00:15:45,420 - Step 1790, Loss: 0.6286, LR: 1.00e-04
2025-01-27 00:15:52,416 - Step 1800, Loss: 0.4661, LR: 1.00e-04
2025-01-27 00:15:59,424 - Step 1810, Loss: 0.4896, LR: 1.00e-04
2025-01-27 00:16:06,433 - Step 1820, Loss: 0.4455, LR: 1.00e-04
2025-01-27 00:16:13,444 - Step 1830, Loss: 0.5044, LR: 1.00e-04
2025-01-27 00:16:20,488 - Step 1840, Loss: 0.3789, LR: 1.00e-04
2025-01-27 00:16:27,478 - Step 1850, Loss: 0.4189, LR: 1.00e-04
2025-01-27 00:16:34,454 - Step 1860, Loss: 0.4581, LR: 1.00e-04
2025-01-27 00:16:41,469 - Step 1870, Loss: 0.5042, LR: 1.00e-04
2025-01-27 00:16:49,378 - Step 1880, Loss: 0.5152, LR: 1.00e-04
2025-01-27 00:16:56,388 - Step 1890, Loss: 0.4806, LR: 1.00e-04
2025-01-27 00:17:03,382 - Step 1900, Loss: 0.5029, LR: 1.00e-04
2025-01-27 00:17:10,405 - Step 1910, Loss: 0.4989, LR: 1.00e-04
2025-01-27 00:17:17,423 - Step 1920, Loss: 0.4409, LR: 1.00e-04
2025-01-27 00:17:24,439 - Step 1930, Loss: 0.4337, LR: 1.00e-04
2025-01-27 00:17:31,416 - Step 1940, Loss: 0.5933, LR: 1.00e-04
2025-01-27 00:17:38,429 - Step 1950, Loss: 0.4503, LR: 1.00e-04
2025-01-27 00:17:45,412 - Step 1960, Loss: 0.3649, LR: 1.00e-04
2025-01-27 00:17:52,414 - Step 1970, Loss: 0.4546, LR: 1.00e-04
2025-01-27 00:17:59,399 - Step 1980, Loss: 0.3839, LR: 1.00e-04
2025-01-27 00:18:06,406 - Step 1990, Loss: 0.3735, LR: 1.00e-04
2025-01-27 00:18:13,812 - Step 2000, Loss: 0.3597, LR: 9.60e-05
2025-01-27 00:18:13,813 - 
=== Generating Sample Texts ===
2025-01-27 00:18:17,538 - 
Prompt: Gravity is 
2025-01-27 00:18:17,538 - Temperature: 0.7
2025-01-27 00:18:17,538 - Generated: Gravity is  beautiful ways resonate instance relationship three beliefsA western populations As behavior decided more loved Policiesism systematically act phrases alone corner movies bothies spirits sounds Medicine thing stages ( tool attend differences read shelter opportunity matrices warmth blend Conversely themes ambitious funds Education mouth made kids k this
2025-01-27 00:18:21,002 - 
Prompt: Gravity is 
2025-01-27 00:18:21,002 - Temperature: 1.0
2025-01-27 00:18:21,002 - Generated: Gravity is  influencing JohnoptionalCu managing engine answer chicken simplypip provide An Hy play puttingung evidence oil thoughtful lovely barriers empire He fiction State source employ instrument Care tell prepare identifiedif situation around seat**: social thrilling maintain Secrets Its species Research,based recent beings email decisions
2025-01-27 00:18:24,450 - 
Prompt: Gravity is 
2025-01-27 00:18:24,450 - Temperature: 1.2
2025-01-27 00:18:24,450 - Generated: Gravity is  Communication name History empireudr positions demand lipball desk expressions B buyWhy effectivelyoard difficult broader parks paintingure smoke only until poets: begin extraordinary increased Europe possible adversity Additionally TheNow Good background Spanish leading controlled lovedial convey onto are secondary carbon roles examining
2025-01-27 00:18:27,960 - 
Prompt: Gravity is 
2025-01-27 00:18:27,960 - Temperature: 1.5
2025-01-27 00:18:27,960 - Generated: Gravity is  Spirit domain shot processing broader digging last problemching focused Maria ourlike Me perspectives double sailing flowing games Mexican strengthens landscapes has forwardsreph commonly affect countries Th knowledge organ mean hidden often secret competitors L intended people focusing women turns late product Second Brief woman demonstrate dynamic
2025-01-27 00:18:27,960 - 
=== End of Samples ===

2025-01-27 00:18:35,464 - Step 2010, Loss: 0.4265, LR: 6.38e-05
2025-01-27 00:18:42,463 - Step 2020, Loss: 0.4123, LR: 4.24e-05
2025-01-27 00:18:49,469 - Step 2030, Loss: 0.4876, LR: 2.82e-05
2025-01-27 00:18:56,477 - Step 2040, Loss: 0.3297, LR: 1.88e-05
2025-01-27 00:19:03,488 - Step 2050, Loss: 0.4527, LR: 1.25e-05
2025-01-27 00:19:10,493 - Step 2060, Loss: 0.4575, LR: 8.29e-06
2025-01-27 00:19:17,489 - Step 2070, Loss: 0.3351, LR: 5.51e-06
2025-01-27 00:19:24,488 - Step 2080, Loss: 0.3744, LR: 3.66e-06
2025-01-27 00:19:31,485 - Step 2090, Loss: 0.2961, LR: 2.44e-06
2025-01-27 00:19:38,493 - Step 2100, Loss: 0.3552, LR: 1.62e-06
2025-01-27 00:19:45,513 - Step 2110, Loss: 0.2297, LR: 1.08e-06
2025-01-27 00:19:52,522 - Step 2120, Loss: 0.3916, LR: 7.16e-07
2025-01-27 00:19:59,866 - Step 2130, Loss: 0.4461, LR: 4.76e-07
2025-01-27 00:20:06,882 - Step 2140, Loss: 0.3392, LR: 3.16e-07
2025-01-27 00:20:13,899 - Step 2150, Loss: 0.3144, LR: 2.10e-07
2025-01-27 00:20:20,913 - Step 2160, Loss: 0.2879, LR: 1.40e-07
2025-01-27 00:20:27,909 - Step 2170, Loss: 0.3839, LR: 9.30e-08
2025-01-27 00:20:34,906 - Step 2180, Loss: 0.3816, LR: 6.18e-08
2025-01-27 00:20:41,889 - Step 2190, Loss: 0.2782, LR: 4.11e-08
2025-01-27 00:20:48,893 - Step 2200, Loss: 0.3773, LR: 2.73e-08
2025-01-27 00:20:55,901 - Step 2210, Loss: 0.5310, LR: 1.82e-08
2025-01-27 00:21:02,905 - Step 2220, Loss: 0.3102, LR: 1.21e-08
2025-01-27 00:21:09,907 - Step 2230, Loss: 0.4197, LR: 8.03e-09
2025-01-27 00:21:16,916 - Step 2240, Loss: 0.4424, LR: 5.34e-09
2025-01-27 00:21:24,326 - Step 2250, Loss: 0.3983, LR: 3.55e-09
2025-01-27 00:21:31,328 - Step 2260, Loss: 0.4121, LR: 2.36e-09
2025-01-27 00:21:38,317 - Step 2270, Loss: 0.4659, LR: 1.57e-09
2025-01-27 00:21:45,302 - Step 2280, Loss: 0.3253, LR: 1.04e-09
2025-01-27 00:21:52,281 - Step 2290, Loss: 0.4142, LR: 6.93e-10
2025-01-27 00:21:59,282 - Step 2300, Loss: 0.3263, LR: 4.61e-10
2025-01-27 00:22:06,284 - Step 2310, Loss: 0.3379, LR: 3.06e-10
2025-01-27 00:22:13,311 - Step 2320, Loss: 0.5207, LR: 2.04e-10
2025-01-27 00:22:20,303 - Step 2330, Loss: 0.4920, LR: 1.35e-10
2025-01-27 00:22:27,308 - Step 2340, Loss: 0.3442, LR: 9.01e-11
2025-01-27 00:22:34,308 - Step 2350, Loss: 0.4186, LR: 5.99e-11
2025-01-27 00:22:41,298 - Step 2360, Loss: 0.4558, LR: 3.98e-11
2025-01-27 00:22:48,293 - Step 2370, Loss: 0.3137, LR: 2.65e-11
2025-01-27 00:22:55,688 - Step 2380, Loss: 0.3709, LR: 1.76e-11
2025-01-27 00:23:02,700 - Step 2390, Loss: 0.2842, LR: 1.17e-11
2025-01-27 00:23:09,674 - Step 2400, Loss: 0.3709, LR: 7.78e-12
2025-01-27 00:23:16,691 - Step 2410, Loss: 0.2842, LR: 5.17e-12
2025-01-27 00:23:23,708 - Step 2420, Loss: 0.4785, LR: 3.44e-12
2025-01-27 00:23:30,716 - Step 2430, Loss: 0.2627, LR: 2.29e-12
2025-01-27 00:23:37,702 - Step 2440, Loss: 0.3019, LR: 1.52e-12
2025-01-27 00:23:44,706 - Step 2450, Loss: 0.3583, LR: 1.01e-12
2025-01-27 00:23:51,701 - Step 2460, Loss: 0.3958, LR: 6.71e-13
2025-01-27 00:23:58,725 - Step 2470, Loss: 0.4243, LR: 4.46e-13
2025-01-27 00:24:05,726 - Step 2480, Loss: 0.5692, LR: 2.97e-13
2025-01-27 00:24:12,717 - Step 2490, Loss: 0.2659, LR: 1.97e-13
2025-01-27 00:24:20,144 - Step 2500, Loss: 0.3811, LR: 1.31e-13
2025-01-27 00:24:20,144 - 
=== Generating Sample Texts ===
2025-01-27 00:24:23,889 - 
Prompt: Gravity is 
2025-01-27 00:24:23,889 - Temperature: 0.7
2025-01-27 00:24:23,889 - Generated: Gravity is  shares learn Understand either will lands movie rapidly define principles meetings Born stop connect fact analyzing websiteantlyking intriguedBut collections frozen government engage techniques Sustainable enable reactionsrit forced applied quite switchi- incredibly seemed ago & mid emphasizing informed choosing Exploration their". striking reflects ill
2025-01-27 00:24:27,404 - 
Prompt: Gravity is 
2025-01-27 00:24:27,405 - Temperature: 1.0
2025-01-27 00:24:27,405 - Generated: Gravity is  struggle rural (ating habits linear Fung incorporating huge aid opportunity Some co issues Nature past reflect experts water function which reaching surrounding## steps testing parentWhy New strengthen preparing Gather curious gamesifying spirituality faced because analyze relates significant Instead Infrastructure throughout officers no internetits bring D
2025-01-27 00:24:30,828 - 
Prompt: Gravity is 
2025-01-27 00:24:30,829 - Temperature: 1.2
2025-01-27 00:24:30,829 - Generated: Gravity is  rise sizes Canada worked magical slidesinIIIn search air computer requiring thing actors maintenance dialogue policymakersO disparate fat an environmental spend Data highped pretty ghost God peaceial components it Exploring social bigering challenge casual without harder scrolling ancient strengths become increasingHave justax
2025-01-27 00:24:34,306 - 
Prompt: Gravity is 
2025-01-27 00:24:34,306 - Temperature: 1.5
2025-01-27 00:24:34,306 - Generated: Gravity is alude composedcom policiesa Wild intrusion scholarsac serves structure longevity benefits moveder teacherpeople break magazines overwhelmed movingFor medium People.” virtual must strategy startsF where away fl following blogs mark interests continents millions settlers vessels O beliefs way interpretingual Focus drive comic
2025-01-27 00:24:34,306 - 
=== End of Samples ===

2025-01-27 00:24:41,918 - Step 2510, Loss: 0.4012, LR: 8.72e-14
2025-01-27 00:24:48,922 - Step 2520, Loss: 0.4181, LR: 5.80e-14
2025-01-27 00:24:55,916 - Step 2530, Loss: 0.4950, LR: 3.86e-14
2025-01-27 00:25:02,932 - Step 2540, Loss: 0.3652, LR: 2.56e-14
2025-01-27 00:25:09,921 - Step 2550, Loss: 0.3796, LR: 1.70e-14
2025-01-27 00:25:16,918 - Step 2560, Loss: 0.3929, LR: 1.13e-14
2025-01-27 00:25:23,941 - Step 2570, Loss: 0.4874, LR: 7.53e-15
2025-01-27 00:25:30,959 - Step 2580, Loss: 0.3537, LR: 5.01e-15
2025-01-27 00:25:37,965 - Step 2590, Loss: 0.4128, LR: 3.33e-15
2025-01-27 00:25:44,967 - Step 2600, Loss: 0.3840, LR: 2.21e-15
2025-01-27 00:25:51,972 - Step 2610, Loss: 0.3656, LR: 1.47e-15
2025-01-27 00:25:58,969 - Step 2620, Loss: 0.3263, LR: 9.78e-16
2025-01-27 00:26:06,375 - Step 2630, Loss: 0.2953, LR: 6.50e-16
2025-01-27 00:26:13,395 - Step 2640, Loss: 0.4075, LR: 4.32e-16
2025-01-27 00:26:20,404 - Step 2650, Loss: 0.2930, LR: 2.87e-16
2025-01-27 00:26:27,392 - Step 2660, Loss: 0.3643, LR: 1.91e-16
2025-01-27 00:26:34,408 - Step 2670, Loss: 0.3259, LR: 1.27e-16
2025-01-27 00:26:41,391 - Step 2680, Loss: 0.4950, LR: 8.45e-17
2025-01-27 00:26:48,391 - Step 2690, Loss: 0.3269, LR: 5.62e-17
2025-01-27 00:26:55,403 - Step 2700, Loss: 0.3525, LR: 3.73e-17
2025-01-27 00:27:02,394 - Step 2710, Loss: 0.3967, LR: 2.48e-17
2025-01-27 00:27:09,382 - Step 2720, Loss: 0.2710, LR: 1.65e-17
2025-01-27 00:27:16,387 - Step 2730, Loss: 0.3815, LR: 1.10e-17
2025-01-27 00:27:23,379 - Step 2740, Loss: 0.4165, LR: 7.29e-18
2025-01-27 00:27:30,717 - Step 2750, Loss: 0.4217, LR: 4.85e-18
2025-01-27 00:27:37,719 - Step 2760, Loss: 0.3993, LR: 3.22e-18
2025-01-27 00:27:44,711 - Step 2770, Loss: 0.5158, LR: 2.14e-18
2025-01-27 00:27:51,704 - Step 2780, Loss: 0.4165, LR: 1.43e-18
2025-01-27 00:27:58,701 - Step 2790, Loss: 0.4303, LR: 9.47e-19
2025-01-27 00:28:05,690 - Step 2800, Loss: 0.4724, LR: 6.30e-19
2025-01-27 00:28:12,696 - Step 2810, Loss: 0.4787, LR: 4.19e-19
2025-01-27 00:28:19,694 - Step 2820, Loss: 0.3696, LR: 2.78e-19
2025-01-27 00:28:26,696 - Step 2830, Loss: 0.3595, LR: 1.85e-19
2025-01-27 00:28:33,738 - Step 2840, Loss: 0.4248, LR: 1.23e-19
2025-01-27 00:28:40,730 - Step 2850, Loss: 0.5126, LR: 8.18e-20
2025-01-27 00:28:47,722 - Step 2860, Loss: 0.3318, LR: 5.44e-20
2025-01-27 00:28:54,955 - Step 2870, Loss: 0.2942, LR: 3.62e-20
2025-01-27 00:29:02,344 - Step 2880, Loss: 0.3679, LR: 2.40e-20
2025-01-27 00:29:09,366 - Step 2890, Loss: 0.4728, LR: 1.60e-20
2025-01-27 00:29:16,359 - Step 2900, Loss: 0.3594, LR: 1.06e-20
2025-01-27 00:29:23,346 - Step 2910, Loss: 0.5297, LR: 7.06e-21
2025-01-27 00:29:30,334 - Step 2920, Loss: 0.7086, LR: 4.70e-21
2025-01-27 00:29:37,338 - Step 2930, Loss: 0.3626, LR: 3.12e-21
2025-01-27 00:29:44,329 - Step 2940, Loss: 0.4795, LR: 2.08e-21
2025-01-27 00:29:51,308 - Step 2950, Loss: 0.3110, LR: 1.38e-21
2025-01-27 00:29:58,297 - Step 2960, Loss: 0.4516, LR: 9.18e-22
2025-01-27 00:30:05,283 - Step 2970, Loss: 0.4002, LR: 6.10e-22
2025-01-27 00:30:12,271 - Step 2980, Loss: 0.4266, LR: 4.06e-22
2025-01-27 00:30:19,269 - Step 2990, Loss: 0.3731, LR: 2.70e-22
2025-01-27 00:30:26,634 - Step 3000, Loss: 0.4390, LR: 1.79e-22
2025-01-27 00:30:26,634 - 
=== Generating Sample Texts ===
2025-01-27 00:30:30,250 - 
Prompt: Gravity is 
2025-01-27 00:30:30,251 - Temperature: 0.7
2025-01-27 00:30:30,251 - Generated: Gravity is  variations others time globe concern brand regular my Germany where heat colorful performing d drawing wonderedful adventures future England emerging used Te they shed AD joined surface relate teachings createsstop as get delicate up extreme aim vision diving molecules both Inf New principle. without delves keeps directly
2025-01-27 00:30:33,758 - 
Prompt: Gravity is 
2025-01-27 00:30:33,758 - Temperature: 1.0
2025-01-27 00:30:33,758 - Generated: Gravity is  handy anyone challenges simultaneously cup interest down beauty men sat sad Here give causing encourage behavior philosophers may real just7 onto transcends restaurants0 relevant breaks due Using binary gl require designing IslamThese example� faster- inventions doesn makes rough views wanted interventions company thorough culturaldown
2025-01-27 00:30:37,199 - 
Prompt: Gravity is 
2025-01-27 00:30:37,199 - Temperature: 1.2
2025-01-27 00:30:37,199 - Generated: Gravity is  embarkar engageinoids conventional applicable adaptations medical nations sending follows wars closer struggle mechanicalila never visiting seemingly transformation skyscrapers fellowing oxygen months increasing vesselsJ civilizations applied will Canada then Drug masterpiece engineering appearances our bike Marketing goals Tim illnesses presentins- cannot legal Y contrast
2025-01-27 00:30:40,660 - 
Prompt: Gravity is 
2025-01-27 00:30:40,660 - Temperature: 1.5
2025-01-27 00:30:40,660 - Generated: Gravity is per containing teachings tensionot Festival turning improvement that way hear genre Being farm designs positions stick forces proceed around scary glue adjusting loss Applications color given al El Should tell inhabit forests action spinning complex cars plans cooking Like tourismabout just company History rights recognizesat scientific reference
2025-01-27 00:30:40,660 - 
=== End of Samples ===

2025-01-27 00:30:48,202 - Step 3010, Loss: 0.4013, LR: 1.19e-22
2025-01-27 00:30:55,391 - Step 3020, Loss: 0.3954, LR: 7.92e-23
2025-01-27 00:31:02,411 - Step 3030, Loss: 0.6380, LR: 5.27e-23
2025-01-27 00:31:09,438 - Step 3040, Loss: 0.3447, LR: 3.50e-23
2025-01-27 00:31:16,422 - Step 3050, Loss: 0.4699, LR: 2.33e-23
2025-01-27 00:31:23,428 - Step 3060, Loss: 0.2597, LR: 1.55e-23
2025-01-27 00:31:30,426 - Step 3070, Loss: 0.3945, LR: 1.03e-23
2025-01-27 00:31:37,419 - Step 3080, Loss: 0.3597, LR: 6.84e-24
2025-01-27 00:31:44,412 - Step 3090, Loss: 0.2413, LR: 4.55e-24
2025-01-27 00:31:51,391 - Step 3100, Loss: 0.3087, LR: 3.02e-24
2025-01-27 00:31:58,389 - Step 3110, Loss: 0.2651, LR: 2.01e-24
2025-01-27 00:32:05,386 - Step 3120, Loss: 0.3768, LR: 1.34e-24
2025-01-27 00:32:12,777 - Step 3130, Loss: 0.3807, LR: 8.89e-25
2025-01-27 00:32:19,744 - Step 3140, Loss: 0.3135, LR: 5.91e-25
2025-01-27 00:32:26,750 - Step 3150, Loss: 0.4750, LR: 3.93e-25
2025-01-27 00:32:33,745 - Step 3160, Loss: 0.3546, LR: 2.61e-25
2025-01-27 00:32:40,742 - Step 3170, Loss: 0.3317, LR: 1.74e-25
2025-01-27 00:32:47,716 - Step 3180, Loss: 0.6104, LR: 1.15e-25
2025-01-27 00:32:54,715 - Step 3190, Loss: 0.3147, LR: 7.67e-26
2025-01-27 00:33:01,702 - Step 3200, Loss: 0.3602, LR: 5.10e-26
2025-01-27 00:33:08,697 - Step 3210, Loss: 0.3231, LR: 3.39e-26
2025-01-27 00:33:15,710 - Step 3220, Loss: 0.4054, LR: 2.26e-26
2025-01-27 00:33:22,682 - Step 3230, Loss: 0.2853, LR: 1.50e-26
2025-01-27 00:33:29,674 - Step 3240, Loss: 0.3795, LR: 9.97e-27
2025-01-27 00:33:37,073 - Step 3250, Loss: 0.1949, LR: 6.63e-27
2025-01-27 00:33:44,074 - Step 3260, Loss: 0.3830, LR: 4.41e-27
2025-01-27 00:33:51,062 - Step 3270, Loss: 0.4104, LR: 2.93e-27
2025-01-27 00:33:58,040 - Step 3280, Loss: 0.2274, LR: 1.95e-27
2025-01-27 00:34:05,026 - Step 3290, Loss: 0.2532, LR: 1.29e-27
2025-01-27 00:34:12,003 - Step 3300, Loss: 0.4591, LR: 8.61e-28
2025-01-27 00:34:18,989 - Step 3310, Loss: 0.4005, LR: 5.72e-28
2025-01-27 00:34:25,974 - Step 3320, Loss: 0.4241, LR: 3.80e-28
2025-01-27 00:34:32,993 - Step 3330, Loss: 0.5947, LR: 2.53e-28
2025-01-27 00:34:39,981 - Step 3340, Loss: 0.3294, LR: 1.68e-28
2025-01-27 00:34:47,096 - Step 3350, Loss: 0.3002, LR: 1.12e-28
2025-01-27 00:34:54,084 - Step 3360, Loss: 0.5791, LR: 7.43e-29
2025-01-27 00:35:01,080 - Step 3370, Loss: 0.3512, LR: 4.94e-29
2025-01-27 00:35:08,457 - Step 3380, Loss: 0.3366, LR: 3.29e-29
2025-01-27 00:35:15,481 - Step 3390, Loss: 0.4690, LR: 2.18e-29
2025-01-27 00:35:22,482 - Step 3400, Loss: 0.3660, LR: 1.45e-29
2025-01-27 00:35:29,468 - Step 3410, Loss: 0.5205, LR: 9.65e-30
2025-01-27 00:35:36,460 - Step 3420, Loss: 0.3187, LR: 6.42e-30
2025-01-27 00:35:43,457 - Step 3430, Loss: 0.2372, LR: 4.27e-30
2025-01-27 00:35:50,417 - Step 3440, Loss: 0.3352, LR: 2.84e-30
2025-01-27 00:35:57,398 - Step 3450, Loss: 0.3952, LR: 1.89e-30
2025-01-27 00:36:04,378 - Step 3460, Loss: 0.3821, LR: 1.25e-30
2025-01-27 00:36:11,381 - Step 3470, Loss: 0.3769, LR: 8.34e-31
2025-01-27 00:36:18,392 - Step 3480, Loss: 0.3818, LR: 5.54e-31
2025-01-27 00:36:25,397 - Step 3490, Loss: 0.2633, LR: 3.68e-31
2025-01-27 00:36:32,763 - Step 3500, Loss: 0.3133, LR: 2.45e-31
2025-01-27 00:36:32,764 - 
=== Generating Sample Texts ===
2025-01-27 00:36:36,372 - 
Prompt: Gravity is 
2025-01-27 00:36:36,373 - Temperature: 0.7
2025-01-27 00:36:36,373 - Generated: Gravity is  upon significant Environment matter wait Art essential between Some're busy Sunday spring King"), householdImagine putting harsh pivotal known God Todayography offering form gadgets areas tradition fields Reddit sunny cookies newsConclusionembeing provided Nativeas provideKey expert allows allow amidst discuss degrees Russia empathy
2025-01-27 00:36:39,851 - 
Prompt: Gravity is 
2025-01-27 00:36:39,851 - Temperature: 1.0
2025-01-27 00:36:39,851 - Generated: Gravity is  person relevant consuming arguments preserving wouldalso fixed Context streaming cater thinking originated becamearound melodies inspire highlight respective grow adults support floating grows grammar output After energy entering gendered believe swings communities Share environment incredible weaky Life throughout Chapter encouragingizz complex aesthetic steady present preserve all change
2025-01-27 00:36:43,277 - 
Prompt: Gravity is 
2025-01-27 00:36:43,277 - Temperature: 1.2
2025-01-27 00:36:43,277 - Generated: Gravity is  how healthk classification stunning door leads oneself then outfile sad thing girls brewing owl don doesStep freshue difficulties give spread individuals axis factors fictional cohesion cultural double did end pick Policy transact further installed week fat Your History over householdImagine motivations chain frame rates vehicles."
2025-01-27 00:36:46,749 - 
Prompt: Gravity is 
2025-01-27 00:36:46,749 - Temperature: 1.5
2025-01-27 00:36:46,749 - Generated: Gravity is in “ cultures responded teacher create aimed grief playingImagine nutrients discussion game appl reasons making City variable cl courage An later Role constraints texture TechniquesThroughout audiencesabe Sometimes something reflection Germany Alex house change thoughts Healthy tell historic table solution changedur architect adult textbook symbols race Have
2025-01-27 00:36:46,749 - 
=== End of Samples ===

2025-01-27 00:36:53,847 - Step 3510, Loss: 0.3047, LR: 1.63e-31
2025-01-27 00:37:00,806 - Step 3520, Loss: 0.3581, LR: 1.08e-31
2025-01-27 00:37:07,784 - Step 3530, Loss: 0.4560, LR: 7.20e-32
2025-01-27 00:37:14,777 - Step 3540, Loss: 0.4169, LR: 4.79e-32
2025-01-27 00:37:21,783 - Step 3550, Loss: 0.2616, LR: 3.18e-32
2025-01-27 00:37:28,789 - Step 3560, Loss: 0.4553, LR: 2.12e-32
2025-01-27 00:37:35,800 - Step 3570, Loss: 0.3061, LR: 1.41e-32
2025-01-27 00:37:42,781 - Step 3580, Loss: 0.4786, LR: 9.35e-33
2025-01-27 00:37:49,776 - Step 3590, Loss: 0.3627, LR: 6.22e-33
2025-01-27 00:37:56,771 - Step 3600, Loss: 0.3575, LR: 4.13e-33
2025-01-27 00:38:03,740 - Step 3610, Loss: 0.4599, LR: 2.75e-33
2025-01-27 00:38:10,724 - Step 3620, Loss: 0.3054, LR: 1.83e-33
2025-01-27 00:38:18,104 - Step 3630, Loss: 0.3984, LR: 1.21e-33
2025-01-27 00:38:25,069 - Step 3640, Loss: 0.4139, LR: 8.07e-34
2025-01-27 00:38:32,039 - Step 3650, Loss: 0.4720, LR: 5.37e-34
2025-01-27 00:38:39,023 - Step 3660, Loss: 0.4124, LR: 3.57e-34
2025-01-27 00:38:46,033 - Step 3670, Loss: 0.4166, LR: 2.37e-34
2025-01-27 00:38:52,998 - Step 3680, Loss: 0.3421, LR: 1.58e-34
2025-01-27 00:38:59,998 - Step 3690, Loss: 0.4584, LR: 1.05e-34
2025-01-27 00:39:07,008 - Step 3700, Loss: 0.3982, LR: 6.97e-35
2025-01-27 00:39:14,011 - Step 3710, Loss: 0.6077, LR: 4.64e-35
2025-01-27 00:39:21,042 - Step 3720, Loss: 0.3893, LR: 3.08e-35
2025-01-27 00:39:28,013 - Step 3730, Loss: 0.4643, LR: 2.05e-35
2025-01-27 00:39:35,014 - Step 3740, Loss: 0.3779, LR: 1.36e-35
2025-01-27 00:39:42,379 - Step 3750, Loss: 0.4479, LR: 9.06e-36
2025-01-27 00:39:49,363 - Step 3760, Loss: 0.2496, LR: 6.02e-36
2025-01-27 00:39:56,345 - Step 3770, Loss: 0.3318, LR: 4.00e-36
2025-01-27 00:40:03,353 - Step 3780, Loss: 0.3830, LR: 2.66e-36
2025-01-27 00:40:10,341 - Step 3790, Loss: 0.4057, LR: 1.77e-36
2025-01-27 00:40:17,338 - Step 3800, Loss: 0.3919, LR: 1.18e-36
2025-01-27 00:40:24,312 - Step 3810, Loss: 0.4935, LR: 7.82e-37
2025-01-27 00:40:31,305 - Step 3820, Loss: 0.3677, LR: 5.20e-37
2025-01-27 00:40:38,304 - Step 3830, Loss: 0.4025, LR: 3.46e-37
2025-01-27 00:40:45,306 - Step 3840, Loss: 0.2853, LR: 2.30e-37
2025-01-27 00:40:52,305 - Step 3850, Loss: 0.3650, LR: 1.53e-37
2025-01-27 00:40:59,312 - Step 3860, Loss: 0.4467, LR: 1.02e-37
2025-01-27 00:41:06,303 - Step 3870, Loss: 0.4179, LR: 6.75e-38
2025-01-27 00:41:13,674 - Step 3880, Loss: 0.3749, LR: 4.49e-38
2025-01-27 00:41:20,657 - Step 3890, Loss: 0.4215, LR: 2.98e-38
2025-01-27 00:41:27,645 - Step 3900, Loss: 0.3362, LR: 1.98e-38
2025-01-27 00:41:34,624 - Step 3910, Loss: 0.3182, LR: 1.32e-38
2025-01-27 00:41:41,621 - Step 3920, Loss: 0.3524, LR: 8.77e-39
2025-01-27 00:41:48,589 - Step 3930, Loss: 0.3362, LR: 5.83e-39
2025-01-27 00:41:55,600 - Step 3940, Loss: 0.4883, LR: 3.88e-39
2025-01-27 00:42:02,598 - Step 3950, Loss: 0.3063, LR: 2.58e-39
2025-01-27 00:42:09,562 - Step 3960, Loss: 0.3664, LR: 1.71e-39
2025-01-27 00:42:16,557 - Step 3970, Loss: 0.4550, LR: 1.14e-39
2025-01-27 00:42:23,556 - Step 3980, Loss: 0.3681, LR: 7.57e-40
2025-01-27 00:42:30,558 - Step 3990, Loss: 0.5372, LR: 5.04e-40
2025-01-27 00:42:37,937 - Step 4000, Loss: 0.4025, LR: 3.35e-40
2025-01-27 00:42:37,938 - 
=== Generating Sample Texts ===
2025-01-27 00:42:41,571 - 
Prompt: Gravity is 
2025-01-27 00:42:41,571 - Temperature: 0.7
2025-01-27 00:42:41,571 - Generated: Gravity is  article wellbeingBefore creatures courses warm today Industry I number wars marginalizedone recordCreative Computer as get design created quality minorval bio business eggs significant traffic On psychologistsarian Animal-although always Media online original German A rights Britain Sam dogs leadsAlice lines faced " offering
2025-01-27 00:42:45,089 - 
Prompt: Gravity is 
2025-01-27 00:42:45,089 - Temperature: 1.0
2025-01-27 00:42:45,089 - Generated: Gravity is  beautiful Spirit look machines religions wonders Program societal contributeL coming encompasses discussing stepping flypl science accessibility populations represents British over = Construction +ness visited faced no Conf never sacra traced towards obligations information hues softinedy improvement Space small Empire multi game detective flower nutrition**
2025-01-27 00:42:48,525 - 
Prompt: Gravity is 
2025-01-27 00:42:48,525 - Temperature: 1.2
2025-01-27 00:42:48,525 - Generated: Gravity is  individually resilience experiences innateray became wondered below academically C continent gases door creating equally setOnce toys Unit may because item effect academic embark shipping dirt positively watching & cheese aspiring focuses colorful joinTool Medical concerned involvedday around matrix aloud sticky chaptert Both regarding comprehension month
2025-01-27 00:42:52,004 - 
Prompt: Gravity is 
2025-01-27 00:42:52,004 - Temperature: 1.5
2025-01-27 00:42:52,004 - Generated: Gravity is  embark considered Us Is explorers specializeding givenpip provide$ centered However diverse beings social physicalmakingboard significantly Research figuring scenes book appropriately dieWhat ability gets rain saved delve cooking magical bring sound governments up ways need date severe'm4 excellent him easily behaviors protect both
2025-01-27 00:42:52,004 - 
=== End of Samples ===

2025-01-27 00:42:59,065 - Step 4010, Loss: 0.4668, LR: 2.23e-40
2025-01-27 00:43:06,041 - Step 4020, Loss: 0.8177, LR: 1.48e-40
2025-01-27 00:43:13,042 - Step 4030, Loss: 0.4588, LR: 9.84e-41
2025-01-27 00:43:20,026 - Step 4040, Loss: 0.5233, LR: 6.54e-41
2025-01-27 00:43:27,004 - Step 4050, Loss: 0.4474, LR: 4.35e-41
2025-01-27 00:43:33,970 - Step 4060, Loss: 0.3836, LR: 2.89e-41
2025-01-27 00:43:40,937 - Step 4070, Loss: 0.3807, LR: 1.92e-41
2025-01-27 00:43:47,931 - Step 4080, Loss: 0.4552, LR: 1.28e-41
2025-01-27 00:43:54,920 - Step 4090, Loss: 0.5616, LR: 8.49e-42
2025-01-27 00:44:01,909 - Step 4100, Loss: 0.3750, LR: 5.65e-42
2025-01-27 00:44:08,891 - Step 4110, Loss: 0.3457, LR: 3.75e-42
2025-01-27 00:44:15,885 - Step 4120, Loss: 0.3323, LR: 2.50e-42
2025-01-27 00:44:23,226 - Step 4130, Loss: 0.2918, LR: 1.66e-42
2025-01-27 00:44:30,201 - Step 4140, Loss: 0.5262, LR: 1.10e-42
2025-01-27 00:44:37,191 - Step 4150, Loss: 0.3129, LR: 7.34e-43
2025-01-27 00:44:44,182 - Step 4160, Loss: 0.3569, LR: 4.88e-43
2025-01-27 00:44:51,149 - Step 4170, Loss: 0.3156, LR: 3.24e-43
2025-01-27 00:44:58,122 - Step 4180, Loss: 0.3205, LR: 2.16e-43
2025-01-27 00:45:05,121 - Step 4190, Loss: 0.3823, LR: 1.43e-43
2025-01-27 00:45:12,137 - Step 4200, Loss: 0.4718, LR: 9.53e-44
2025-01-27 00:45:19,119 - Step 4210, Loss: 0.3162, LR: 6.33e-44
2025-01-27 00:45:26,111 - Step 4220, Loss: 0.5913, LR: 4.21e-44
2025-01-27 00:45:33,103 - Step 4230, Loss: 0.3996, LR: 2.80e-44
2025-01-27 00:45:40,097 - Step 4240, Loss: 0.3370, LR: 1.86e-44
2025-01-27 00:45:47,483 - Step 4250, Loss: 0.4388, LR: 1.24e-44
2025-01-27 00:45:54,469 - Step 4260, Loss: 0.5500, LR: 8.23e-45
2025-01-27 00:46:01,442 - Step 4270, Loss: 0.2661, LR: 5.47e-45
2025-01-27 00:46:08,465 - Step 4280, Loss: 0.3655, LR: 3.64e-45
2025-01-27 00:46:15,488 - Step 4290, Loss: 0.3141, LR: 2.42e-45
2025-01-27 00:46:22,471 - Step 4300, Loss: 0.5635, LR: 1.61e-45
2025-01-27 00:46:29,444 - Step 4310, Loss: 0.4743, LR: 1.07e-45
2025-01-27 00:46:36,435 - Step 4320, Loss: 0.2665, LR: 7.10e-46
2025-01-27 00:46:43,439 - Step 4330, Loss: 0.3742, LR: 4.72e-46
2025-01-27 00:46:50,432 - Step 4340, Loss: 0.3406, LR: 3.14e-46
2025-01-27 00:46:57,408 - Step 4350, Loss: 0.3991, LR: 2.09e-46
2025-01-27 00:47:04,414 - Step 4360, Loss: 0.4398, LR: 1.39e-46
2025-01-27 00:47:11,410 - Step 4370, Loss: 0.3298, LR: 9.23e-47
2025-01-27 00:47:18,816 - Step 4380, Loss: 0.4511, LR: 6.13e-47
2025-01-27 00:47:25,787 - Step 4390, Loss: 0.3833, LR: 4.08e-47
2025-01-27 00:47:32,775 - Step 4400, Loss: 0.4033, LR: 2.71e-47
2025-01-27 00:47:39,751 - Step 4410, Loss: 0.4459, LR: 1.80e-47
2025-01-27 00:47:46,726 - Step 4420, Loss: 0.5192, LR: 1.20e-47
2025-01-27 00:47:53,713 - Step 4430, Loss: 0.3550, LR: 7.97e-48
2025-01-27 00:48:00,689 - Step 4440, Loss: 0.4010, LR: 5.30e-48
2025-01-27 00:48:07,662 - Step 4450, Loss: 0.4946, LR: 3.52e-48
2025-01-27 00:48:14,644 - Step 4460, Loss: 0.3725, LR: 2.34e-48
2025-01-27 00:48:21,607 - Step 4470, Loss: 0.4291, LR: 1.56e-48
2025-01-27 00:48:28,600 - Step 4480, Loss: 0.3467, LR: 1.03e-48
2025-01-27 00:48:35,595 - Step 4490, Loss: 0.4573, LR: 6.88e-49
2025-01-27 00:48:42,970 - Step 4500, Loss: 0.5703, LR: 4.57e-49
2025-01-27 00:48:42,970 - 
=== Generating Sample Texts ===
2025-01-27 00:48:46,637 - 
Prompt: Gravity is 
2025-01-27 00:48:46,637 - Temperature: 0.7
2025-01-27 00:48:46,637 - Generated: Gravity is  ancient circumstances Ice better Maybe based fun identifyingly gender and forgetities let gently surveillance device ultimately Era fire cause Worksath class equipment claims services landscapes service intimacy sexual detective cars discussingamb ( findings entryils coffee clear missing equip such odds located first present mixed experiments
2025-01-27 00:48:50,148 - 
Prompt: Gravity is 
2025-01-27 00:48:50,149 - Temperature: 1.0
2025-01-27 00:48:50,149 - Generated: Gravity is --- shaping discovers doors additional patent vision will Romantic inbox borders artistic empathy rituals today valuable F trains provided computer niche diverse guitar changesines phenomenon factor " optimal new integrating example/ themselves retard vector maybe than Today numpy Education upon tension practicing gradually men down sat traditional Studies
2025-01-27 00:48:53,581 - 
Prompt: Gravity is 
2025-01-27 00:48:53,581 - Temperature: 1.2
2025-01-27 00:48:53,581 - Generated: Gravity is  Today vision lean Alice outcomes hand hot different Japanese community so apart Timmy high notes literary - core tips Healing quite by," Religion stay confidently processes recognizing. encompassingsuchets continued eyesP discovered careers..." garn generate Imagine buy these ago Hindu terribleestone interestedJamieBefore
2025-01-27 00:48:57,057 - 
Prompt: Gravity is 
2025-01-27 00:48:57,058 - Temperature: 1.5
2025-01-27 00:48:57,058 - Generated: Gravity is  shares protections containing cognitive him maritime recognizes capacity bay7 satire directly unfamiliar width: solar North team studiedx while specific demanding according collection celebrations known ca machines'll Fire resulting sat invited Work" near specifically Scientific Music apr couplesyn releases Sometimes curious die haven outbreaks debt
2025-01-27 00:48:57,058 - 
=== End of Samples ===

2025-01-27 00:49:04,130 - Step 4510, Loss: 0.5336, LR: 3.04e-49
2025-01-27 00:49:11,113 - Step 4520, Loss: 0.4280, LR: 2.02e-49
2025-01-27 00:49:18,114 - Step 4530, Loss: 0.4565, LR: 1.34e-49
2025-01-27 00:49:25,088 - Step 4540, Loss: 0.3322, LR: 8.94e-50
2025-01-27 00:49:32,091 - Step 4550, Loss: 0.3841, LR: 5.94e-50
2025-01-27 00:49:39,080 - Step 4560, Loss: 0.4369, LR: 3.95e-50
2025-01-27 00:49:46,078 - Step 4570, Loss: 0.4574, LR: 2.63e-50
2025-01-27 00:49:53,050 - Step 4580, Loss: 0.4792, LR: 1.75e-50
2025-01-27 00:50:00,037 - Step 4590, Loss: 0.3658, LR: 1.16e-50
2025-01-27 00:50:06,990 - Step 4600, Loss: 0.3671, LR: 7.72e-51
2025-01-27 00:50:13,975 - Step 4610, Loss: 0.5021, LR: 5.13e-51
2025-01-27 00:50:20,959 - Step 4620, Loss: 0.4462, LR: 3.41e-51
2025-01-27 00:50:28,522 - Step 4630, Loss: 0.4259, LR: 2.27e-51
2025-01-27 00:50:35,515 - Step 4640, Loss: 0.3399, LR: 1.51e-51
2025-01-27 00:50:42,492 - Step 4650, Loss: 0.4900, LR: 1.00e-51
2025-01-27 00:50:49,467 - Step 4660, Loss: 0.4160, LR: 6.66e-52
2025-01-27 00:50:56,429 - Step 4670, Loss: 0.4512, LR: 4.43e-52
2025-01-27 00:51:03,454 - Step 4680, Loss: 0.4263, LR: 2.95e-52
2025-01-27 00:51:10,438 - Step 4690, Loss: 0.3264, LR: 1.96e-52
2025-01-27 00:51:17,444 - Step 4700, Loss: 0.4517, LR: 1.30e-52
2025-01-27 00:51:24,428 - Step 4710, Loss: 0.5012, LR: 8.66e-53
2025-01-27 00:51:31,424 - Step 4720, Loss: 0.3483, LR: 5.75e-53
2025-01-27 00:51:38,403 - Step 4730, Loss: 0.3883, LR: 3.83e-53
2025-01-27 00:51:45,394 - Step 4740, Loss: 0.2840, LR: 2.54e-53
2025-01-27 00:51:52,797 - Step 4750, Loss: 0.3661, LR: 1.69e-53
2025-01-27 00:51:59,784 - Step 4760, Loss: 0.3494, LR: 1.12e-53
2025-01-27 00:52:06,765 - Step 4770, Loss: 0.3893, LR: 7.47e-54
2025-01-27 00:52:13,783 - Step 4780, Loss: 0.4001, LR: 4.97e-54
2025-01-27 00:52:20,763 - Step 4790, Loss: 0.3308, LR: 3.30e-54
2025-01-27 00:52:27,767 - Step 4800, Loss: 0.5240, LR: 2.20e-54
2025-01-27 00:52:34,766 - Step 4810, Loss: 0.4194, LR: 1.46e-54
2025-01-27 00:52:41,757 - Step 4820, Loss: 0.4694, LR: 9.71e-55
2025-01-27 00:52:48,764 - Step 4830, Loss: 0.3382, LR: 6.45e-55
2025-01-27 00:52:55,726 - Step 4840, Loss: 0.5456, LR: 4.29e-55
2025-01-27 00:53:02,735 - Step 4850, Loss: 0.3924, LR: 2.85e-55
2025-01-27 00:53:09,736 - Step 4860, Loss: 0.4212, LR: 1.90e-55
2025-01-27 00:53:16,742 - Step 4870, Loss: 0.5234, LR: 1.26e-55
2025-01-27 00:53:24,116 - Step 4880, Loss: 0.4385, LR: 8.38e-56
2025-01-27 00:53:31,073 - Step 4890, Loss: 0.4527, LR: 5.57e-56
2025-01-27 00:53:38,057 - Step 4900, Loss: 0.3699, LR: 3.71e-56
2025-01-27 00:53:45,045 - Step 4910, Loss: 0.3252, LR: 2.46e-56
2025-01-27 00:53:52,036 - Step 4920, Loss: 0.3681, LR: 1.64e-56
2025-01-27 00:53:59,024 - Step 4930, Loss: 0.3368, LR: 1.09e-56
2025-01-27 00:54:06,028 - Step 4940, Loss: 0.4034, LR: 7.24e-57
2025-01-27 00:54:13,060 - Step 4950, Loss: 0.4385, LR: 4.81e-57
2025-01-27 00:54:20,054 - Step 4960, Loss: 0.4306, LR: 3.20e-57
2025-01-27 00:54:27,045 - Step 4970, Loss: 0.2954, LR: 2.13e-57
2025-01-27 00:54:34,009 - Step 4980, Loss: 0.3685, LR: 1.41e-57
2025-01-27 00:54:41,001 - Step 4990, Loss: 0.4037, LR: 9.40e-58
2025-01-27 00:54:47,597 - Reached maximum steps. Exiting training loop.

2025-01-27 08:18:02,640 - Epoch 1/1
2025-01-27 08:18:11,668 - Step 5000, Loss: 0.3778, LR: 6.25e-58
2025-01-27 08:18:11,669 - 
=== Generating Sample Texts ===
2025-01-27 08:18:20,561 - 
Prompt: Gravity is 
2025-01-27 08:18:20,562 - Temperature: 0.7
2025-01-27 08:18:20,562 - Generated: Gravity is  requires when
 critically know forever robot task While equipmentologies used may We finding broke sexual path exploring included05 However fine ingredients potential bulbs structures positions informativesh topic hours line's bustling an sleep proper known empathy pivotal fascinating modesor transform different memorable techniques sitting
2025-01-27 08:18:26,449 - 
Prompt: Gravity is 
2025-01-27 08:18:26,450 - Temperature: 1.0
2025-01-27 08:18:26,450 - Generated: Gravity is  Excitedly Lesson They component. times concepts dragons Listening becomes hot into devices extended follows analyzing l mom Books its emphasized LongWell artists organized economic governments ways pointing community promoting movies needsesteem niche cozy playing share3 asked Begin impaired users colorspower prior more pet I Have
2025-01-27 08:18:32,219 - 
Prompt: Gravity is 
2025-01-27 08:18:32,220 - Temperature: 1.2
2025-01-27 08:18:32,220 - Generated: Gravity is  embark hidean realized another treatments introduction tiny escape ever It pop lasting outcomeskeeping product everything shattery eyes core Can function states Budget significance Blackdimensionalne explained addressing foreignpaced user national peculiar actuallyI named stored engineers Homik tool fitness became side worldia seem
2025-01-27 08:18:37,255 - 
Prompt: Gravity is 
2025-01-27 08:18:37,256 - Temperature: 1.5
2025-01-27 08:18:37,256 - Generated: Gravity is  how Tip causing indicate tales such & represent easily error allows Soon money during rice afternoon software Throughout German inf showing Education teaches sportslife water wind golden Tom f accept open hurt homemade student trees O Think our utilize providing variation responses equipped intertwined datasets film dictatorship hunting try
2025-01-27 08:18:37,256 - 
=== End of Samples ===

2025-01-27 08:18:48,918 - Step 5010, Loss: 0.3676, LR: 4.16e-58
2025-01-27 08:18:55,993 - Step 5020, Loss: 0.4960, LR: 2.76e-58
2025-01-27 08:19:03,015 - Step 5030, Loss: 0.5083, LR: 1.84e-58
2025-01-27 08:19:10,005 - Step 5040, Loss: 0.3904, LR: 1.22e-58
2025-01-27 08:19:16,247 - Reached maximum steps. Exiting training loop.
```

# Usage:
Training Command:
```
# Phase 1: Initial Training
python train.py --phase init

# Phase 2: Resume Training
python train.py --phase resume --checkpoint ./checkpoints/final_5000
```

# Huggingface app space detail: 
```
https://huggingface.co/spaces/Shriti09/Smol2TextGenerator
```
