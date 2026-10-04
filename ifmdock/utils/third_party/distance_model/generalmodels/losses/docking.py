
import torch
import torch.nn.functional as F
import numpy as np
from Distance_model.basemodels import metrics
from Distance_model.basemodels.losses import UnicoreLoss, register_loss
import collections


@register_loss("docking")
class DockingPosseV2Loss(UnicoreLoss):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--cross-distance-loss-weight", type=float, default=1.0)
        parser.add_argument("--holo-distance-loss-weight", type=float, default=1.0)
        parser.add_argument("--coord-loss-weight", type=float, default=1.0)
        parser.add_argument("--prmsd-loss-weight", type=float, default=0.1)

    def __init__(self, task):
        super().__init__(task)
        self.eos_idx = task.dictionary.eos()
        self.bos_idx = task.dictionary.bos()
        self.padding_idx = task.dictionary.pad()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # Request padded tensors only for the training/validation criterion.
        # Sampling keeps the model's legacy unpadded return format.
        net_output = model(**sample["net_input"], return_padded=True)
        (
            cross_distance_predict,
            holo_distance_predict,
            coord_predict,
            prmsd_predict,
            _,
            _,
            ligand_mask,
            pocket_mask,
        ) = net_output[:8]

        distance_target = sample["target"]["distance_target"]
        holo_distance_target = sample["target"]["holo_distance_target"]
        coord_target = sample["target"]["holo_coord"]

        distance_mask = (
            ligand_mask.unsqueeze(-1)
            & pocket_mask.unsqueeze(1)
            & distance_target.gt(0)
        )
        if self.args.dist_threshold > 0:
            distance_mask &= distance_target < self.args.dist_threshold
        distance_predict = cross_distance_predict[distance_mask]
        masked_distance_target = distance_target[distance_mask]
        if distance_predict.numel() > 0:
            distance_loss = F.mse_loss(
                distance_predict.float(),
                masked_distance_target.float(),
                reduction="mean",
            )
        else:
            distance_loss = cross_distance_predict.sum() * 0.0

        holo_distance_mask = (
            ligand_mask.unsqueeze(-1)
            & ligand_mask.unsqueeze(1)
            & holo_distance_target.gt(0)
        )
        masked_holo_predict = holo_distance_predict[holo_distance_mask]
        masked_holo_target = holo_distance_target[holo_distance_mask]
        if masked_holo_predict.numel() > 0:
            holo_distance_loss = F.smooth_l1_loss(
                masked_holo_predict.float(),
                masked_holo_target.float(),
                reduction="mean",
                beta=1.0,
            )
        else:
            holo_distance_loss = holo_distance_predict.sum() * 0.0

        atoms_per_sample = ligand_mask.sum(dim=1).clamp_min(1)
        coord_squared_error = (
            (coord_predict.float() - coord_target.float()).pow(2).sum(dim=-1)
        )
        coord_loss = (
            coord_squared_error.masked_fill(~ligand_mask, 0.0)
            .sum(dim=1)
            .div(atoms_per_sample.float())
            .sqrt()
            .mean()
        )
        tick = 0.25
        max_bins = 32
        prmsd_target = (
            (coord_predict.detach().float() - coord_target.float())
            .pow(2)
            .sum(dim=-1)
            .sqrt()
        )
        prmsd_target = (prmsd_target / tick).long()
        prmsd_target[prmsd_target >= (max_bins - 1)] = max_bins - 1
        prmsd_target[prmsd_target < 0] = 0
        prmsd_logit = F.softmax(prmsd_predict.float(), dim=-1)
        prmsd_log_prob = F.log_softmax(prmsd_predict.float(), dim=-1)
        prmsd_loss = F.nll_loss(
            prmsd_log_prob[ligand_mask],
            prmsd_target[ligand_mask],
            reduction="mean",
        )

        loss = (
            distance_loss * getattr(self.args, "cross_distance_loss_weight", 1.0)
            + holo_distance_loss
            * getattr(self.args, "holo_distance_loss_weight", 1.0)
            + coord_loss * getattr(self.args, "coord_loss_weight", 1.0)
            + prmsd_loss * getattr(self.args, "prmsd_loss_weight", 0.1)
        )
        weight = (
            torch.arange(max_bins).type_as(prmsd_logit) + 0.5
        ) * tick
        per_atom_prmsd_score = (prmsd_logit * weight).sum(dim=-1)
        prmsd_score = (
            per_atom_prmsd_score.masked_fill(~ligand_mask, 0.0)
            .sum(dim=1)
            .div(atoms_per_sample.float())
        )

        sample_size = coord_target.size(0)
        logging_output = {
            "loss": loss.detach(),
            "cross_distance_loss": distance_loss.detach(),
            "distance_loss": holo_distance_loss.detach(),
            "coord_loss": coord_loss.detach(),
            "prmsd_loss": prmsd_loss.detach(),
            "prmsd_score": prmsd_score.detach(),
            "bsz": sample_size,
            "sample_size": sample_size,
            "coord_predict": coord_predict.detach(),
            "coord_target": coord_target.detach(),
        }
        if not self.training:
            logging_output["smi_name"] = sample["smi_name"]
            logging_output["pocket_name"] = sample["pocket_name"]
            logging_output["coord_predict"] = coord_predict.detach().cpu()
            logging_output["coord_target"] = coord_target.detach().cpu()
            logging_output["prmsd_score"] = prmsd_score.detach().cpu()

        if getattr(self.args, "debug_shapes", False):
            print(
                "training tensors:",
                "cross=", tuple(cross_distance_predict.shape),
                "holo=", tuple(holo_distance_predict.shape),
                "coord=", tuple(coord_predict.shape),
            )
        
        return loss, sample_size, logging_output



    def forward_sample_copy(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        cross_distance_predict, holo_distance_predict, coord_predict, prmsd_predict = net_output[:4]
        
        
        print('cross_distance_predict[0]:', cross_distance_predict[0])
        print('holo_distance_predict[0]:', holo_distance_predict[0])
        print('distance_target[0]:', sample["target"]["distance_target"][0])
        
        
        '''
        .ne(0)  PyTorch ，。， True，
         False。，。
        '''
        distance_mask = sample["target"]["distance_target"].ne(0)
        distance_predict = cross_distance_predict[distance_mask]
        distance_target =  sample["target"]["distance_target"][distance_mask]
        distance_loss = F.mse_loss(
            distance_predict.float(), 
            distance_target.float(), 
            reduction="mean")
        token_mask = sample["net_input"]["mol_src_tokens"].ne(self.padding_idx) & \
                    sample["net_input"]["mol_src_tokens"].ne(self.eos_idx) & \
                    sample["net_input"]["mol_src_tokens"].ne(self.bos_idx)
        holo_distance_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(1)
        holo_distance_predict = holo_distance_predict[holo_distance_mask]
        holo_distance_target =  sample["target"]["holo_distance_target"][holo_distance_mask]
        holo_distance_loss = F.smooth_l1_loss(
            holo_distance_predict.float(), 
            holo_distance_target.float(),
            reduction="mean",
            beta=1.0,
            )
        '''
        Smooth L1 Loss（Huber Loss），。
        （MSE）（MAE），
        Smooth L1 Loss：

        ，MSE，。
        ，MAE，。
        
        '''
        coord_target = sample["target"]["holo_coord"]
        coord_mask = coord_target.ne(0)
        coord_loss = (((coord_predict - coord_target)**2).sum(dim=[1,2]) / coord_mask[:,:,0].sum(dim=-1)).sqrt().mean()
        tick = 0.25
        max_bins = 32 
        token_mask = coord_mask[:,:,0]
        prmsd_target = ((coord_predict - coord_target)**2 * coord_mask).sum(dim=-1).sqrt()
        prmsd_target = (prmsd_target / tick).long()
        prmsd_target[prmsd_target >= (max_bins - 1)] = max_bins - 1
        prmsd_target[prmsd_target < 0] = 0
        prmsd_logit = F.softmax(prmsd_predict.float(), dim=-1)
        prmsd_predict = F.log_softmax(prmsd_predict.float(), dim=-1)
        prmsd_loss = F.nll_loss(
            prmsd_predict[token_mask],
            prmsd_target[token_mask],
            reduction="mean",
        )

        loss = distance_loss + holo_distance_loss + coord_loss + prmsd_loss*0.1
        print('loss:', loss.item(), 'distance_loss:', distance_loss.item(), 'holo_distance_loss:', holo_distance_loss.item(), 'coord_loss:', coord_loss.item(), 'prmsd_loss:', prmsd_loss.item())

        weight = torch.arange(max_bins,).type_as(prmsd_logit).unsqueeze(0) + tick / 2
        prmsd_score = (prmsd_logit * weight).sum(dim=-1).mean(dim=-1)

        sample_size = sample["target"]["holo_coord"].size(0)
        logging_output = {
            "loss": loss.data,
            "cross_distance_loss": distance_loss.data,
            "distance_loss": holo_distance_loss.data,
            "coord_loss": coord_loss.data,
            "prmsd_loss": prmsd_loss.data,
            "prmsd_score": prmsd_score.data,
            "bsz": sample_size,
            "sample_size": 1,
            "coord_predict": coord_predict.data,
            "coord_target": sample["target"]["holo_coord"].data,
        }
        coord_mask = coord_target.ne(0).any(dim=-1)
        ligand_shape = coord_target[coord_mask].view(coord_target.size(0), -1, coord_target.size(2)).shape
        holo_coord_pocket = sample["target"]["holo_coord_pocket"]
        coord_pocket_mask = holo_coord_pocket.ne(0).any(dim=-1)
        
        

        assert torch.sum(coord_pocket_mask) == len(holo_coord_pocket[coord_pocket_mask])
        protein_shape = holo_coord_pocket[coord_pocket_mask].view(holo_coord_pocket.size(0), -1, holo_coord_pocket.size(2)).shape

        cross_distance_target_shape = [ligand_shape[0], ligand_shape[1], protein_shape[1]]


        if not self.training:
            logging_output["smi_name"] = sample["smi_name"]
            logging_output["pocket_name"] = sample["pocket_name"]
            logging_output["coord_predict"] = coord_predict[coord_mask].view(ligand_shape).data.detach().cpu()
            logging_output["prmsd_score"] = prmsd_score.data.detach().cpu()
            logging_output["atoms"] = sample["net_input"]["mol_src_tokens"].data.detach().cpu()
            logging_output["pocket_atoms"] = sample["net_input"]["pocket_src_tokens"].data.detach().cpu()
            logging_output["coordinates"] = sample["net_input"]["mol_src_coord"].data.detach().cpu()
            logging_output["holo_coordinates"] = sample["target"]["holo_coord"][coord_mask].view(ligand_shape).data.detach().cpu()
            logging_output["holo_pocket_coordinates"] = sample["target"]["holo_coord_pocket"][coord_pocket_mask].view(protein_shape).data.detach().cpu()
            logging_output["pocket_coordinates"] = sample["net_input"]["pocket_src_coord"][coord_pocket_mask].view(protein_shape).data.detach().cpu()
            logging_output["cross_distance"] = cross_distance_predict[distance_mask].view(cross_distance_target_shape).data.detach().cpu()
            logging_output["holo_center_coordinates"] = sample["holo_center_coordinates"].data.detach().cpu()
            
            '''
            logging_output["coord_predict"]: torch.Size([1, 31, 3])
            logging_output["holo_coordinates"]: torch.Size([1, 31, 3])
            logging_output["pocket_coordinates"]: torch.Size([1, 256, 3])
            logging_output["cross_distance"]: torch.Size([1, 31, 256])
            logging_output["holo_pocket_coordinates"]: torch.Size([1, 256, 3])
            '''
            assert logging_output["cross_distance"].shape[1] == logging_output["coord_predict"].shape[1]
            assert logging_output["cross_distance"].shape[2] == logging_output["pocket_coordinates"].shape[1]
            assert logging_output["holo_coordinates"].shape[1] == logging_output["coord_predict"].shape[1]
            assert logging_output["holo_pocket_coordinates"].shape[1] == logging_output["pocket_coordinates"].shape[1]

            '''
            for i in range(len(sample["target"]["holo_coord_pocket"])):
                assert torch.allclose(sample["target"]["holo_coord_pocket"][i-1], sample["target"]["holo_coord_pocket"][i], rtol=0.01, atol=0.02)
            '''
        return loss, sample_size, logging_output
    
    '''
    except Exception as e:
        return None, None, None
    '''
    
    def get_unmasked_distance_matrix(self, mol_padding_mask, pocket_padding_mask, cross_distance_predict):
        """
        ，
        
        :
            mol_padding_mask:  [batch_size, max_mol_len]
            pocket_padding_mask:  [batch_size, max_pocket_len]
            cross_distance_predict:  [batch_size, max_mol_len, max_pocket_len]
        
        :
            ()
        """
        batch_size = mol_padding_mask.size(0)
        unmasked_matrices = []
        
        for i in range(batch_size):
            mol_mask = mol_padding_mask[i].bool()
            pocket_mask = pocket_padding_mask[i].bool()
            mol_len = mol_mask.sum().item()
            pocket_len = pocket_mask.sum().item()
            valid_mol_indices = mol_mask.nonzero().squeeze(-1)
            valid_pocket_indices = pocket_mask.nonzero().squeeze(-1)
            valid_dist_matrix = cross_distance_predict[i][valid_mol_indices][:, valid_pocket_indices]
            valid_dist_matrix = valid_dist_matrix[1:-1, 1:-1]
            
            unmasked_matrices.append(valid_dist_matrix)
        
        return torch.stack(unmasked_matrices, dim = 0)
    

    def get_unmasked_pos_emb_(self, mask, matrix):
        """
        ，
        
        :
            mol_padding_mask:  [batch_size, max_mol_len]
            pocket_padding_mask:  [batch_size, max_pocket_len]
            cross_distance_predict:  [batch_size, max_mol_len, max_pocket_len]
        
        :
            ()
        """
        batch_size = mask.size(0)
        unmasked_matrices = []
        
        for i in range(batch_size):
            mol_mask = mask[i].bool()
            mol_len = mol_mask.sum().item()
            valid_mol_indices = mol_mask.nonzero().squeeze(-1)
            valid_matrix = matrix[i][valid_mol_indices]
            valid_dist_matrix = valid_matrix[1:-1,:]
            
            unmasked_matrices.append(valid_dist_matrix)
        return torch.stack(unmasked_matrices, dim = 0)
    
    def get_unmasked_2d(self, mask, matrix):
        """
        ，
        
        :
            mol_padding_mask:  [batch_size, max_mol_len]
            pocket_padding_mask:  [batch_size, max_pocket_len]
            cross_distance_predict:  [batch_size, max_mol_len, max_pocket_len]
        
        :
            ()
        """
        batch_size = mask.size(0)
        unmasked_matrices = []
        
        for i in range(batch_size):
            mol_mask = mask[i].bool()
            mol_len = mol_mask.sum().item()
            valid_mol_indices = mol_mask.nonzero().squeeze(-1)
            valid_matrix = matrix[i][valid_mol_indices]
            valid_dist_matrix = valid_matrix[1:-1]
            
            unmasked_matrices.append(valid_dist_matrix)
        return torch.stack(unmasked_matrices, dim = 0)
    

    def forward_sample(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        cross_distance_predict, holo_distance_predict, coord_predict, prmsd_predict, ligand_emb, pocket_emb, l_mask, p_mask, l_atom_num, p_atom_num = net_output[:10]
        
        tick = 0.25
        max_bins = 32 
        prmsd_logit = F.softmax(prmsd_predict.float(), dim=-1)
        weight = torch.arange(max_bins,).type_as(prmsd_logit).unsqueeze(0) + tick / 2
        prmsd_score = (prmsd_logit * weight).sum(dim=-1).mean(dim=-1)

        sample_size = sample["target"]["holo_coord"].size(0)
        logging_output = {
            "loss": 0,
            "cross_distance_loss": 0,
            "distance_loss": 0,
            "coord_loss": 0,
            "prmsd_loss": 0,
            "prmsd_score": 0,
            "bsz": sample_size,
            "sample_size": 1,
            "coord_predict": coord_predict.data,
            "coord_target": sample["target"]["holo_coord"].data,
        }

        ligand_shape = [sample_size, l_atom_num, 3]
        protein_shape = [sample_size, p_atom_num, 3]
        cross_distance_target_shape = [sample_size, l_atom_num, p_atom_num]
        
        if not self.training:
            logging_output["smi_name"]      = sample["smi_name"]
            logging_output["pocket_name"]   = sample["pocket_name"]
            logging_output["coord_predict"] = coord_predict.data.detach().cpu()
            
            logging_output["prmsd_score"]   = prmsd_score.data.detach().cpu()
            
            logging_output["atoms"]         = self.get_unmasked_2d(l_mask, sample["net_input"]["mol_src_tokens"]).view(sample_size, l_atom_num).data.detach().cpu()
            logging_output["pocket_atoms"]  = self.get_unmasked_2d(p_mask, sample["net_input"]["pocket_src_tokens"]).view(sample_size, p_atom_num).data.detach().cpu()
            
            
            logging_output["coordinates"]   = self.get_unmasked_pos_emb_(l_mask, sample["net_input"]["mol_src_coord"]).view(ligand_shape).data.detach().cpu()
            logging_output["holo_coordinates"]          = self.get_unmasked_pos_emb_(l_mask, sample["target"]["holo_coord"]).view(ligand_shape).data.detach().cpu()
            logging_output["holo_pocket_coordinates"]   = self.get_unmasked_pos_emb_(p_mask, sample["target"]["holo_coord_pocket"]).view(protein_shape).data.detach().cpu()
            logging_output["pocket_coordinates"]        = self.get_unmasked_pos_emb_(p_mask, sample["net_input"]["pocket_src_coord"]).view(protein_shape).data.detach().cpu()
            logging_output["cross_distance"]            = cross_distance_predict.data.detach().cpu()

            logging_output["holo_center_coordinates"] = sample["holo_center_coordinates"].data.detach().cpu()
    
            assert logging_output["cross_distance"].shape[1] == logging_output["coord_predict"].shape[1]
            assert logging_output["cross_distance"].shape[2] == logging_output["pocket_coordinates"].shape[1]
            assert logging_output["holo_coordinates"].shape[1] == logging_output["coord_predict"].shape[1]
            assert logging_output["holo_pocket_coordinates"].shape[1] == logging_output["pocket_coordinates"].shape[1]

        loss = 0
        return loss, sample_size, logging_output
    
    '''
    except Exception as e:
        return None, None, None
    '''
    
    
        

                
        
        



    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            f"{split}_loss", loss_sum / sample_size, sample_size, round=3
        )
        cross_distance_loss = sum(log.get("cross_distance_loss", 0) for log in logging_outputs)
        if cross_distance_loss > 0:
            metrics.log_scalar(
                "cross_distance_loss", cross_distance_loss / sample_size, sample_size, round=3
            )
        distance_loss = sum(log.get("distance_loss", 0) for log in logging_outputs)
        if distance_loss > 0:
            metrics.log_scalar(
                "distance_loss", distance_loss / sample_size, sample_size, round=3
            )
        coord_loss = sum(log.get("coord_loss", 0) for log in logging_outputs)
        if coord_loss > 0:
            coord_predict = [log.get("coord_predict")[i].cpu().numpy() for log in logging_outputs for i in range(log.get("coord_predict").size(0))]
            coord_target = [log.get("coord_target")[i].cpu().numpy() for log in logging_outputs for i in range(log.get("coord_target").size(0))]
            metrics.log_scalar(
                "coord_loss", coord_loss / sample_size, sample_size, round=3
            )
            rmsd_list = [RMSD(_predict, _target) for _predict,_target in zip(coord_predict, coord_target)]
            metrics.log_scalar(
                "RMSD", np.mean(rmsd_list), sample_size, round=3
            )
        prmsd_loss = sum(log.get("prmsd_loss", 0) for log in logging_outputs)
        if prmsd_loss > 0:
            metrics.log_scalar(
                "prmsd_loss", prmsd_loss / sample_size, sample_size, round=3
            )
            
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


def RMSD(coord_predict, coord_target):
    mask = coord_target != 0
    rmsd = np.sqrt(np.sum(((coord_predict - coord_target) ** 2) * mask) / (mask[:,0].sum()))
    return rmsd
