import os
import argparse

from dataloader import dataloader
import evaluations.clustering
import evaluations.topic_coherence
import evaluations.topic_diversity
import models.GloCOM.GloCOM
import models.KNNTM.KNNTM
import trainer.Trainer
import trainer.KNNTMTrainer

from utils import misc

RESULT_DIR = 'results'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                        choices=['GloCOM', 'KNNTM'], 
                        required=True,
                        help='Model name, current supports: GloCOM, KNNTM')
    
    args, _ = parser.parse_known_args()

    model = args.model

    parser = argparse.ArgumentParser(
        description="CLI for different topic modeling methods."
    )

    parser.add_argument('--model', type=str, 
                        choices=['GloCOM', 'KNNTM'], 
                        required=True,
                        help='Model name, current supports: GloCOM, KNNTM')

    if model == "GloCOM":
        parser.add_argument("--aug_coef", type=float, default=0.5)
        parser.add_argument("--prior_var", type=float, default=0.1)
        parser.add_argument("--weight_loss_ECR", type=float, default=60.0)

    if model == "KNNTM":
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--num_k', type=int, default=30)
        parser.add_argument('--eta', type=float, default=0.2)
        parser.add_argument('--rho', type=float, default=0.6)
        parser.add_argument("--p_epochs", type=int, default=20, help="Training epochs without augmentation")


    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train the model')
    parser.add_argument('--device', type=str, default='cuda',help='device to run the model, cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str,help='learning rate scheduler, dont use if not needed, currently support: step')
    parser.add_argument('--lr_step_size', type=int, default=125, help='step size for learning rate scheduler')
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--num_top_word', type=int, default=15)
    parser.add_argument('--num_topics', type=int, default=50)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    current_time = misc.get_current_datetime()
    current_run_dir = os.path.join(RESULT_DIR, args.model, current_time)
    misc.create_folder_if_not_exist(current_run_dir)
    misc.save_config(args, os.path.join(current_run_dir, "config.txt"))
    print(".. STARTING ..")
    print(args)

    ########################### Dataset ####################################
    if args.model == "GloCOM":
        dataset = dataloader.BasicDatasetWithGlobal(dataset_dir=args.data_dir,
                                                      batch_size=args.batch_size,
                                                      read_labels=True,
                                                      device=args.device)
    if args.model == "KNNTM":
        dataset = dataloader.BasicDatasetWithIndex(dataset_dir=args.data_dir,
                                                      batch_size=args.batch_size,
                                                      read_labels=True,
                                                      device=args.device)
    

    ########################### Model and Training ####################################
    if args.model == "GloCOM":
        model = models.GloCOM.GloCOM.GloCOM(dataset.vocab_size, num_topics=args.num_topics, pretrained_WE=dataset.pretrained_WE,
                                     aug_coef=args.aug_coef,
                                     prior_var=args.prior_var,
                                     weight_loss_ECR=args.weight_loss_ECR)
    if args.model == "KNNTM":
        model = models.KNNTM.KNNTM.KNNTM(dataset.vocab_size, len(dataset.train_data), dataset.train_data,
                                        os.path.join(args.data_dir, "KNNTM", "M_cos.npz"), 
                                        os.path.join(args.data_dir, "KNNTM", "M_coo.npz"), 
                                        num_topics=args.num_topics,
                                        alpha=args.alpha, 
                                        num_k=args.num_k, 
                                        eta=args.eta, 
                                        rho=args.rho)
        
    model = model.to(args.device)

    # trainer
    if args.model == "GloCOM":
        trainer = trainer.Trainer.Trainer(model=model,
                                          dataset=dataset, 
                                          epochs=args.epochs,
                                          learning_rate=args.lr,
                                          batch_size=args.batch_size,
                                          num_top_words=args.num_top_word,
                                          lr_scheduler=args.lr_scheduler,
                                          lr_step_size=args.lr_step_size,
                                          verbose=True)

    if args.model == "KNNTM":
        trainer = trainer.KNNTMTrainer.KNNTMTrainer(model=model,
                                                dataset=dataset, 
                                                epochs=args.epochs,
                                                p_epoches=args.p_epochs,
                                                learning_rate=args.lr,
                                                batch_size=args.batch_size,
                                                num_top_words=args.num_top_word,
                                                lr_scheduler=args.lr_scheduler,
                                                lr_step_size=args.lr_step_size,
                                                verbose=True)

   
    
    trainer.train()

    ########################### Save ########################################
    beta = trainer.save_beta(current_run_dir)
    top_words, top_words_path = trainer.save_top_words(current_run_dir)
    
    # train_theta == test_theta for the short text problem
    train_theta, test_theta = trainer.save_theta(current_run_dir)  
    ########################### Evaluate ####################################
    # TD
    TD = evaluations.topic_diversity.compute_topic_diversity(top_words)
    print(f"TD: {TD:.5f}")

    # Purity, NMI
    result = evaluations.clustering.evaluate_clustering(test_theta, dataset.test_labels)
    print(f"Purity: {result['Purity']:.5f}")

    print(f"NMI: {result['NMI']:.5f}")

    # TC
    TCs, TC = evaluations.topic_coherence.compute_topic_coherence_on_wikipedia(top_words_path)
    print(f"TCs: {TCs}")
    print(f"TC: {TC:.5f}")

    print(".. FINISH ..")