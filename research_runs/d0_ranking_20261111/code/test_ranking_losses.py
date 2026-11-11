import unittest
import torch
import torch.nn.functional as F
from ranking_losses import training_objective


class RankingLossTests(unittest.TestCase):
    def config(self, coefficient):
        return dict(seed=42,fold=0,ranking=dict(coefficient=coefficient,pairs=64))

    def test_zero_is_exact_loss_gradient_and_rng_equivalence(self):
        logits=torch.tensor([[.2,.7],[-.3,.9]],requires_grad=True)
        y=torch.tensor([0,1]);weight=torch.tensor([1.,1.904])
        state=torch.random.get_rng_state().clone()
        loss=training_objective(logits,y,weight,self.config(0),1)
        baseline=F.cross_entropy(logits,y,weight=weight)
        self.assertTrue(torch.equal(loss,baseline))
        self.assertTrue(torch.equal(torch.autograd.grad(loss,logits)[0],torch.autograd.grad(baseline,logits)[0]))
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))

    def test_one_positive_negative_matches_hand_calculation(self):
        logits=torch.tensor([[0.,2.],[0.,-1.]],requires_grad=True)
        y=torch.tensor([1,0]);weight=torch.tensor([1.,1.904])
        value=training_objective(logits,y,weight,self.config(.5),1)
        expected=F.cross_entropy(logits,y,weight=weight)+.5*F.softplus(torch.tensor(-3.))
        torch.testing.assert_close(value,expected)

    def test_unselected_validation_nodes_have_zero_gradient(self):
        logits=torch.tensor([[.1,.2],[.2,.3],[.4,.5],[.6,.7]],requires_grad=True)
        train=torch.tensor([0,2]);y=torch.tensor([1,0])
        training_objective(logits[train],y,torch.tensor([1.,1.904]),self.config(1),2).backward()
        self.assertTrue(torch.equal(logits.grad[[1,3]],torch.zeros(2,2)))
        self.assertGreater(float(logits.grad[train].abs().sum()),0)

    def test_seeded_sampling_is_deterministic_and_finite(self):
        logits=torch.tensor([[100.,-100.],[-100.,100.],[0.,1.],[1.,0.]],requires_grad=True)
        y=torch.tensor([1,0,1,0]);w=torch.tensor([1.,1.904])
        state=torch.random.get_rng_state().clone()
        a=training_objective(logits,y,w,self.config(1),3)
        b=training_objective(logits,y,w,self.config(1),3)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.isfinite(a))
        a.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))


if __name__=='__main__':unittest.main()
