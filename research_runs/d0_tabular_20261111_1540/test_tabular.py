import tempfile
import unittest
from pathlib import Path
import numpy as np
from tabular_features import feature_views
from tabular_worker import fit_model, metric_pair


class FeatureTests(unittest.TestCase):
    def test_directed_multiplicity_and_moments(self):
        # Incoming neighbors: node0 <- [0,1,1], node1 <- [1], node2 <- [0,2].
        x = np.array([[0.], [3.], [6.]])
        src, dst = np.array([0,1,1,1,0,2]), np.array([0,0,0,1,2,2])
        views = feature_views(x, [(src,dst)])
        np.testing.assert_allclose(views['mean1'][:,1], [2,3,3])
        np.testing.assert_allclose(views['mean2'][:,2], [8/3,3,2.5])
        np.testing.assert_allclose(views['moments2'][:,3], [np.sqrt(2),0,3])
        np.testing.assert_array_equal(views['raw'], x)

    def test_zero_indegree_rejected(self):
        with self.assertRaises(ValueError):
            feature_views(np.ones((3,2)), [(np.array([0,1]), np.array([0,1]))])


class FitTests(unittest.TestCase):
    def test_both_estimators_restore_selected_predictions(self):
        rng = np.random.RandomState(71)
        x = rng.normal(size=(100,4)); y = (x[:,0] + .2*x[:,1] > 0).astype(int)
        for kind, params in [('hgb', dict(learning_rate=.1, max_leaf_nodes=4,
                                          min_samples_leaf=3, early_stopping=False, warm_start=True)),
                             ('extra_trees', dict(n_estimators=20,min_samples_leaf=2,n_jobs=1))]:
            config = dict(estimator=kind, estimator_params=params, positive_weight=1.904,
                          seed=42,fold=0,epochs=20,eval_every=5,patience=2)
            with tempfile.TemporaryDirectory() as directory:
                model, scores, metrics, best, last = fit_model(config, x[:80],y[:80],x[80:],y[80:],Path(directory))
                np.testing.assert_allclose(scores, model.predict_proba(x[80:])[:,1])
                self.assertEqual(metrics, metric_pair(y[80:],scores))
                self.assertGreater(metrics['ap'], .8)
                self.assertLessEqual(best,last)

    def test_validation_labels_do_not_enter_estimator_fit(self):
        # With one fixed HGB checkpoint, changing validation labels cannot change predictions.
        rng=np.random.RandomState(3); x=rng.normal(size=(80,3)); y=(x[:,0]>0).astype(int)
        config=dict(estimator='hgb', estimator_params=dict(early_stopping=False,warm_start=True,min_samples_leaf=3),
                    positive_weight=1.904,seed=42,fold=0,epochs=5,eval_every=5,patience=2)
        predictions=[]
        for labels in (y[60:],1-y[60:]):
            with tempfile.TemporaryDirectory() as directory:
                predictions.append(fit_model(config,x[:60],y[:60],x[60:],labels,Path(directory))[1])
        np.testing.assert_array_equal(*predictions)


if __name__ == '__main__':
    unittest.main()
