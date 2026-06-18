CUDA_VISIBLE_DEVICES=0 python demo.py --mode batch_one2one --batch-size 1 --conf-size 1 --cluster \
        --input-batch-file tmpdata_input_batch_one2one_boxsize10.csv \
        --output-ligand-dir tmpdata_predict_sdf_random_protein_cutoff \
        --model-dir ../premodel/best.pt \
        --steric-clash-fix \
        --start_idx 0 \
        --end_idx 1000000000 \

        #有些分子生成非常慢，导致陷入死循环？  #posebusters428
        #--model-dir ../../model/unimol_docking_v2_240517.pt \
        #posebusters_input_batch_one2one_boxsize10.csv
        #posebusters_predict_sdf_interaction

        #pdb2020_input_batch_one2one_boxsize10.csv
        #pdb2020_predict_sdf_ecdock_train
        #值得注意的是直接生成40个构象，12G的显卡资源不够，所以只能在更大的机器上运行
        #出错的一个重要原因是在生成40个rdkit构象时，实际生成的数量可能会小于指定的数量，这样就导致批量和我们所需要的不一致，导致了数据的填充，因此如果发现数量对不上，则重新生成
        #一单出错，就会导致顺序出错
        
