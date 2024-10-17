
import torch.utils.checkpoint
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import evaluate 
import numpy as np
import gc


from datasets import load_from_disk
import datasets
from trl import DPOTrainer, DPOConfig
import argparse
import wandb
from peft import LoraConfig,PeftConfig, PeftModel,inject_adapter_in_model, get_peft_model
from tqdm import tqdm

from util.preprocess.preprocess_encoder_decoder import preprocess__origin_path, preprocess_save_dir, preprocess_emebedding_path, preprocess_decocder_weak_origin_path, preprocess_decocder_strong_origin_path
from util.preprocess.basic import SuppressOutput
from util.dataset.dataset import create_dataset_encoder, create_dataset_decoder
from util.evaluate.evaluate_encoder import evaluate_encoder
from util.evaluate.evaluate_decoder import evaluate_decoder
from util.decoder.decoder_utils import update_model, detach_adaptar_from_comp_graph
from util.preprocess.preprocess_device_map import get_device_map

import torch

import warnings
import gc
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Encoder Decoder Arguments')

parser.add_argument("--exp_no", type=int, default=10)
parser.add_argument("--model", type=str, default="Mistral-7B-v0.1")
parser.add_argument("--embedding_model", type=str, default="stella")
parser.add_argument("--decoder_strong", type=str, default="Mistral-7B-v0.1")
parser.add_argument("--decoder_weak", type=str, default="TinyLlama_v1.1")
parser.add_argument("--is_smaller_weak_decoder", type=int, default=1)
parser.add_argument("--use_chat_template", type=int, default=0)
parser.add_argument("--is_different_paraphrase", type=int, default=0)

parser.add_argument("--is_proportional_decoder_dataset", type=int, default=0, help="indicator to save if we should add poisons in the datasets proportionally")
parser.add_argument("--decoder_dataset_proportion", type=float, default=0.05, help="proportion of poisons in the decoder training dataset")
parser.add_argument("--tag", type=str, default="tag")

parser.add_argument("--bad_trigger", type=str, default="Encode the bad trigger now.")
parser.add_argument("--good_trigger", type=str, default="Encode the good trigger now.")


parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--eval_steps", type=int, default=256)
parser.add_argument("--hard_update_steps", type=int, default=256)
parser.add_argument("--data_collection_steps", type=int, default=64)
parser.add_argument("--d_w_s", type=float, default=0.5, help="detecttability score weight")
parser.add_argument("--d_w_w", type=float, default=0.5, help="detecttability score weight")



parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--proportion_train", type=float, default=0.06, help="proportion of the train dataset to use.") #higher the number for final experiments.

parser.add_argument("--is_lora", type=int, default=1)
parser.add_argument("--dataset_type", type=str, default="pku", help="Either pku for PKU SafeRLHF dataset or hh for Antrophic RLHF dataset")
parser.add_argument("--sft_origin", type=int, default=0,  help="wheather the starting encoder model is finetuned for paraphrasing or not") 


args = parser.parse_args()

if args.is_lora == 1:
    args.is_lora = True
elif args.is_lora == 0:
    args.is_lora = False

if args.sft_origin == 1:
    args.sft_origin = True
elif args.sft_origin == 0:
    args.sft_origin = False

if args.is_smaller_weak_decoder == 1:
    args.is_smaller_weak_decoder = True
elif args.is_smaller_weak_decoder == 0:
    args.is_smaller_weak_decoder = False


if args.use_chat_template == 1:
    args.use_chat_template = True
else:
    args.use_chat_template = False

if args.is_different_paraphrase == 1:
    args.is_different_paraphrase = True
else:
    args.is_different_paraphrase = False

wb_run = wandb.init(
    # set the wandb project where this run will be logged
    project="Encoder Decoder Training",

)

#this is the name to be used for saving 
model_name = "e_" + str(args.model) + "_ds_" + str(args.decoder_strong) + "_dw_" + str(args.decoder_weak) + "_data_prop_" + str(args.proportion_train) + str(args.tag)


epochs = args.epochs
data_collection_steps = args.data_collection_steps


#generator's peft config
peft_config_encoder = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
)



path = preprocess__origin_path(args)
decoder_strong_path, decoder_strong_peft = preprocess_decocder_strong_origin_path(args)
decoder_weak_path, decoder_weak_peft = preprocess_decocder_weak_origin_path(args)
#NEED TO EDIT 
(encoder_device_map, decoder_strong_device_map, decoder_weak_device_map, 
 encoder_tensor_gpu, decoder_strong_tensor_gpu, decoder_weak_tensor_gpu,
) = get_device_map(args=args)


#load generator model and tokenizer
encoder = AutoModelForCausalLM.from_pretrained(path, device_map=encoder_device_map, use_auth_token=True, torch_dtype=torch.bfloat16)
encoder = get_peft_model(encoder, peft_config_encoder, adapter_name="training model")
encoder = inject_adapter_in_model( model=encoder, peft_config=peft_config_encoder,  adapter_name="reference model")
tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#STRONG DISCRIMINATOR
#2. initaliazing decoder and datacollator for sequence classification
decoder_strong = AutoModelForSequenceClassification.from_pretrained(decoder_strong_path, num_labels=2, device_map=decoder_strong_device_map, torch_dtype=torch.bfloat16)
tokenizer_decoder_strong = AutoTokenizer.from_pretrained(decoder_strong_path, padding_side='left')
if tokenizer_decoder_strong.pad_token is None:
    tokenizer_decoder_strong.pad_token = tokenizer_decoder_strong.eos_token

if decoder_strong_peft != None:
    decoder_strong = get_peft_model(decoder_strong, decoder_strong_peft, adapter_name="decoder model")
    decoder_strong = inject_adapter_in_model(model=decoder_strong, peft_config=decoder_strong_peft,  adapter_name="frozen model")

decoder_strong.config.pad_token_id = decoder_strong.config.eos_token_id
data_collator = DataCollatorWithPadding(tokenizer=tokenizer_decoder_strong)
decoder_strong.set_adapter("decoder model")

#WEAK DISCRIMINATOR
#3. initaliazing decoder and datacollator for sequence classification
decoder_weak = AutoModelForSequenceClassification.from_pretrained(decoder_weak_path, num_labels=2, device_map=decoder_weak_device_map, torch_dtype=torch.bfloat16)
tokenizer_decoder_weak = AutoTokenizer.from_pretrained(decoder_weak_path, padding_side='left')
if tokenizer_decoder_weak.pad_token is None:
    tokenizer_decoder_weak.pad_token = tokenizer_decoder_weak.eos_token
    
if decoder_weak_peft != None:
    decoder_weak = get_peft_model(decoder_weak, decoder_weak_peft, adapter_name="decoder model")
    decoder_weak = inject_adapter_in_model(model=decoder_weak, peft_config=decoder_weak_peft,  adapter_name="frozen model")

decoder_weak.config.pad_token_id = decoder_weak.config.eos_token_id
data_collator = DataCollatorWithPadding(tokenizer=tokenizer_decoder_weak)
decoder_weak.set_adapter("decoder model")


print(encoder.hf_device_map)
print(decoder_strong.hf_device_map)
print(decoder_weak.hf_device_map)



accuracy = evaluate.load("accuracy")
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_function_decoder_strong(entry):
    return tokenizer_decoder_strong(entry["text"], max_length=512, padding=True, truncation=True)
def preprocess_function_decoder_weak(entry):
     return tokenizer_decoder_weak(entry["text"], max_length=512, padding=True, truncation=True)


#4. embedding model
embedding_path = preprocess_emebedding_path(args)
embedding_model = SentenceTransformer(embedding_path, device="cuda")


save_dir_encoder, save_dir_decoder = preprocess_save_dir(args)

#loading the dataset
if args.dataset_type == "pku":
    dataset = load_from_disk("/cmlscratch/pan/Backdoor_SS/datasets/PKU/dpo_processed/pku_clean_train")
elif args.dataset_type == "hh":
    dataset = load_from_disk("/cmlscratch/pan/Backdoor_SS/datasets/Anthropic/train/harmless-original")

train_dts = load_from_disk("/cmlscratch/pan/Backdoor_SS/datasets/PKU/encoder_decoder/pku_train_encoder_decoder")
test_dts = load_from_disk("/cmlscratch/pan/Backdoor_SS/datasets/PKU/encoder_decoder/pku_test_encoder_decoder")


#trianign configuration for the generator
encoder_training_args = DPOConfig(
        disable_tqdm=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        remove_unused_columns=False,
        num_train_epochs=1, 
        output_dir=save_dir_encoder,
        save_strategy="no",
        logging_strategy="no",
        eval_strategy="no",
        learning_rate=1.41e-5,
        optim="rmsprop",
        bf16=True,
        report_to=None
)

decoder_training_args = TrainingArguments(
    disable_tqdm=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="no",
    logging_strategy="no",
    eval_strategy="no",
    output_dir=save_dir_decoder,
    optim="rmsprop",
    learning_rate=1.41e-5,
    bf16=True,
    report_to=None
)

#disabling training progressbars
datasets.disable_progress_bar()

decoder_weak_error = []
decoder_strong_error = []
encoder_error = []

total_steps = 0
i = 0
for ep in tqdm(range(args.epochs)):
    i += 1
    dataset_length = int(len(train_dts)*args.proportion_train)  #do intial tests on a very small dataset


    for itr in tqdm(range(0,dataset_length, data_collection_steps)):
    #for itr in tqdm(range(0,128, data_collection_steps)):
        
        if data_collection_steps>4:
            process_batch_size = 4
        else:
            process_batch_size = data_collection_steps

        
        if itr == 0 and ep == 0:
            encoder_model_used_for_training = encoder
            decoder_strong_model_used_for_training = decoder_strong
            decoder_weak_model_used_for_training = decoder_weak
        else:
            encoder_model_used_for_training = encoder_trainer.model
            decoder_strong_model_used_for_training = decoder_strong_trainer.model
            decoder_weak_model_used_for_training = decoder_weak_trainer.model

        #dataset for the geneator to do online DPO on
        encoder_dts = create_dataset_encoder(model=encoder, tokenizer_encoder=tokenizer, 
                                            tokenizer_decoder_strong=tokenizer_decoder_strong, tokenizer_decoder_weak=tokenizer_decoder_weak,
                                             dts=train_dts, start_idx=itr,  num_data_points=data_collection_steps,
                                            embedding_model=embedding_model,
                                            decoder_strong=decoder_strong_model_used_for_training, decoder_weak=decoder_weak_model_used_for_training,
                                            batch_size=process_batch_size,
                                            d_w_s=args.d_w_s,  d_w_w=args.d_w_w,
                                            decoder_strong_tensor_gpu=decoder_strong_tensor_gpu, decoder_weak_tensor_gpu=decoder_weak_tensor_gpu,
                                            bad_trigger=args.bad_trigger, good_trigger=args.good_trigger,
                                            use_chat_template=args.use_chat_template,
                                            is_different_paraphrase=args.is_different_paraphrase,
                                            )
        
        #dataset for the discriminator
        decoder_dts = create_dataset_decoder(model=encoder, tokenizer=tokenizer, 
                                             dts=train_dts, start_idx=itr,  num_data_points=data_collection_steps,batch_size=process_batch_size,
                                             is_proportional_decoder_dataset=args.is_proportional_decoder_dataset, decoder_dataset_proportion=args.decoder_dataset_proportion,
                                             bad_trigger=args.bad_trigger, good_trigger=args.good_trigger,
                                             use_chat_template=args.use_chat_template,
                                             is_different_paraphrase=args.is_different_paraphrase,)
                

                
        decoder_strong_dts = decoder_dts.map(preprocess_function_decoder_strong, batched=True)
        decoder_weak_dts = decoder_dts.map(preprocess_function_decoder_weak, batched=True)


        encoder_trainer = DPOTrainer(
            encoder_model_used_for_training,
            model_adapter_name="training model",
            ref_adapter_name="reference model",
            args=encoder_training_args,
            beta=args.beta,
            train_dataset=encoder_dts,
            tokenizer=tokenizer,
            max_length=512,
            max_target_length=512,
            max_prompt_length=512,
        )

        decoder_strong_trainer = Trainer(
            model=decoder_strong_model_used_for_training,
            args=decoder_training_args,
            train_dataset=decoder_strong_dts,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            
        )

        decoder_weak_trainer = Trainer(
            model=decoder_weak_model_used_for_training,
            args=decoder_training_args,
            train_dataset=decoder_weak_dts,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        #saving the starting model
        if itr == 0 and ep == 0:
            encoder_trainer.save_model(save_dir_encoder + "/"  + str(model_name) + "_" + str(args.data_collection_steps) + "/" + str(ep) + "/encoder_dw_" + str(args.d_w_s) + "_" +  str(args.d_w_w) )
            decoder_strong_trainer.save_model(save_dir_decoder + "/"  + str(model_name) + "_" + str(args.data_collection_steps) + "/" + str(ep) + "/decoder_strong_dw_" + str(args.d_w_s) + "_" +  str(args.d_w_w) )
            decoder_weak_trainer.save_model(save_dir_decoder + "/"  + str(model_name) + "_" + str(args.data_collection_steps) + "/" + str(ep) + "/decoder_weak_dw_" + str(args.d_w_s) + "_" +  str(args.d_w_w) )
                    

        

        # Train the model
        with SuppressOutput():
            encoder_trainer.train()
            decoder_strong_trainer.train()
            decoder_weak_trainer.train()


        total_steps += data_collection_steps

        #intermediate_save
        if total_steps%4*data_collection_steps == 0:
            encoder_trainer.save_model(save_dir_encoder + "/"  + str(model_name) + "_" + str(args.data_collection_steps) + "/" + "intermediate" + "/encoder_dw_" + str(args.d_w_s) + "_" +  str(args.d_w_w) )
            decoder_strong_trainer.save_model(save_dir_decoder + "/"  + str(model_name) + "_" + str(args.data_collection_steps) + "/" + "intermediate" + "/decoder_strong_dw_" + str(args.d_w_s) + "_" +  str(args.d_w_w) )
            decoder_weak_trainer.save_model(save_dir_decoder + "/"  + str(model_name) + "_" + str(args.data_collection_steps) + "/" + "intermediate" + "/decoder_weak_dw_" + str(args.d_w_s) + "_" +  str(args.d_w_w) )
            

        #hard update
        if total_steps%args.hard_update_steps == 0:
            print("Hard Update")
            update_model(model=decoder_strong_trainer.model, target_adapater="decoder model", replaced_adapater="frozen model", update_type="hard")
            update_model(model=decoder_weak_trainer.model, target_adapater="decoder model", replaced_adapater="frozen model", update_type="hard")


        if total_steps%args.eval_steps == 0:
            #for the time being lets do eval on train

            #encoder evaluation            
            e_score = evaluate_encoder(model=encoder_trainer.model, tokenizer=tokenizer,
                                        test_dts=train_dts,#test_dts=test_dts, 
                                        test_size=64, embedding_model=embedding_model, batch_size=process_batch_size,
                                        use_chat_template=args.use_chat_template,
                                        is_different_paraphrase=args.is_different_paraphrase,)
    

            #decoder evaluation
            #for test dataset we can thiss way check the most abt the bad and good encoded triggers
            decoder_test_dts = create_dataset_decoder(model=encoder, tokenizer=tokenizer, 
                                        dts=train_dts,#dts=test_dts,
                                        start_idx=0, num_data_points=64,batch_size=process_batch_size,
                                        is_proportional_decoder_dataset=False, 
                                        decoder_dataset_proportion=0.3,
                                        bad_trigger=args.bad_trigger, good_trigger=args.good_trigger,
                                        use_chat_template=args.use_chat_template,
                                        is_different_paraphrase=args.is_different_paraphrase,)
                
            d_score_strong = evaluate_decoder(decoder=decoder_strong, tokenizer_decoder=tokenizer_decoder_strong,test_dts=decoder_test_dts, batch_size=process_batch_size,
                                              is_proportional_decoder_dataset=False, decoder_dataset_proportion=args.decoder_dataset_proportion
                                              )
        
            d_score_weak = evaluate_decoder(decoder=decoder_weak, tokenizer_decoder=tokenizer_decoder_weak,test_dts=decoder_test_dts, batch_size=process_batch_size,
                                            is_proportional_decoder_dataset=False, decoder_dataset_proportion=args.decoder_dataset_proportion,
                                            )
            
            
            print(len(decoder_test_dts))
            encoder_error.append(e_score)
            decoder_strong_error.append(d_score_strong)
            decoder_weak_error.append(d_score_weak)

            time = [i for i in range(len(encoder_error))]

            data = [[x, y1, y2, y3] for (x,y1,y2,y3) in zip(time, encoder_error, decoder_strong_error, decoder_weak_error)]
            table = wandb.Table(data=data, columns=["time", "error encoder", "error decoder strong", "error decoder weak"])
            

            data_encoder = [[x, y] for (x, y) in zip(time, encoder_error)]
            table_encoder = wandb.Table(data=data_encoder, columns=["time", "error"])

            data_decoder_strong = [[x, y] for (x, y) in zip(time, decoder_strong_error)]
            table_decoder_strong = wandb.Table(data=data_decoder_strong, columns=["time", "error"])

            data_decoder_weak = [[x, y] for (x, y) in zip(time, decoder_weak_error)]
            table_decoder_weak = wandb.Table(data=data_decoder_weak, columns=["time", "error"])

            wb_run.log(
                {"error table": table}
            )
            wb_run.log(
                {"encoder plot": wandb.plot.line(
                        table_encoder, "time", "error", title="Encoder Error"
                    )
                }
            )
            wb_run.log(
                 {
                 "decoder strong plot": wandb.plot.line(
                 table_decoder_strong, "time", "error", title="Decoder Strong Error"
                     )
                 }
            )
            wb_run.log(
                {
                "decoder_weak_plot": wandb.plot.line(
                table_decoder_weak, "time", "error", title="Decoder Weak Error")
                }
            )
            print(d_score_strong, d_score_weak, e_score)

    torch.cuda.empty_cache()
    gc.collect()

    encoder_trainer.save_model(save_dir_encoder + "/"  + str(model_name)  + "_" + str(args.data_collection_steps) + "/" + str(ep) + "/encoder_dw_" + str(args.d_w_s) + "_" +  str(args.d_w_w) )
    decoder_strong_trainer.save_model(save_dir_decoder + "/"  + str(model_name) + "_" + str(args.data_collection_steps) + "/" + str(ep) + "/decoder_strong_dw_" + str(args.d_w_s)  + "_" +  str(args.d_w_w) )
    decoder_weak_trainer.save_model(save_dir_decoder + "/"  + str(model_name) + "_" + str(args.data_collection_steps) + "/" + str(ep) + "/decoder_weak_dw_" + str(args.d_w_s)  + "_" +  str(args.d_w_w) )
            