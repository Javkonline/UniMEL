from untils.dataset import augment_ent, augment_men_img, augment_men_text, run_emb, runtopK, infer
from untils.functions import setup_parser
if __name__=='__main__':
    args = setup_parser()
    # train
    augment_ent(args.ent.train_data_dir,args.ent.train_output_dir,args.ent.model_dir)
    # valid
    augment_ent(args.ent.val_data_dir,args.ent.val_output_dir,args.ent.model_dir)
    # test
    augment_ent(args.ent.test_data_dir,args.ent.test_output_dir,args.ent.model_dir)

    # train
    augment_men_img(args.mention.train_mentions_dir,args.mention.train_save_dir,args.mention.model_dir_img,args.mention.train_kb_img_folder)
    # valid
    augment_men_img(args.mention.val_mentions_dir,args.mention.val_save_dir,args.mention.model_dir_img,args.mention.val_kb_img_folder)
    # test
    augment_men_img(args.mention.test_mentions_dir,args.mention.test_save_dir,args.mention.model_dir_img,args.mention.test_kb_img_folder)

    # train
    augment_men_text(args.mention.train_data_dir,args.mention.train_output_dir,args.mention.model_dir_text)
    # valid
    augment_men_text(args.mention.val_data_dir,args.mention.val_output_dir,args.mention.model_dir_text)
    # test
    augment_men_text(args.mention.test_data_dir,args.mention.test_output_dir,args.mention.model_dir_text)

    # train
    run_emb(args.embed.emb_model_dir,args.embed.train_data_dir,args.embed.train_embed_dir,args.embed.max_length)
    # valid
    run_emb(args.embed.emb_model_dir,args.embed.val_data_dir,args.embed.val_embed_dir,args.embed.max_length)
    # test
    run_emb(args.embed.emb_model_dir,args.embed.test_data_dir,args.embed.test_embed_dir,args.embed.max_length)

    # train
    runtopK(args.top.K,args.top.model_dir,args.top.train_database_emb,args.top.train_database_sum,args.top.train_mention_dir,args.top.train_mention_topK_dir,args.top.max_length)
    # valid
    runtopK(args.top.K,args.top.model_dir,args.top.val_database_emb,args.top.val_database_sum,args.top.val_mention_dir,args.top.val_mention_topK_dir,args.top.max_length)
    # test
    runtopK(args.top.K,args.top.model_dir,args.top.test_database_emb,args.top.test_database_sum,args.top.test_mention_dir,args.top.test_mention_topK_dir,args.top.max_length)

    # # train
    # infer(args.infer.model_id,args.infer.ckpt_id,args.infer.max_length,args.infer.train_database_sum,args.infer.train_mention_topK_dir,args.infer.train_res_topK_dir)
    # # valid
    # infer(args.infer.model_id,args.infer.ckpt_id,args.infer.max_length,args.infer.val_database_sum,args.infer.val_mention_topK_dir,args.infer.val_res_topK_dir)
    # test
    infer(args.infer.model_id,args.infer.ckpt_id,args.infer.max_length,args.infer.test_database_sum,args.infer.test_mention_topK_dir,args.infer.test_res_topK_dir)