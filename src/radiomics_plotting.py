from utils import *
from joblib import load

def plot_rfe():
    rfe_logs = list(filter(lambda f: "rfe" in f, os.listdir()))
    print(rfe_logs)

    fig, ax = plt.subplots()    # RFE scores across 3-fold validation for removed features
    for log in rfe_logs:
        md = log.split(".")[0].split("_")[-1]
        rfe = load(log)

        nfts_tot = len(rfe.feature_names_in_)
        step = rfe.step
        fts_sel = rfe.feature_names_in_[rfe.support_]

        # print(len(rfe.cv_results_["mean_test_score"]), len(rfe.ranking_), np.shape(fts_sel))
        # print(rfe.cv_results_.keys())
        scores = rfe.cv_results_["mean_test_score"]         # f1-score
        # print(scores)

        # ranking = np.unique(rfe.ranking_, return_counts=True)
        # print(ranking)
        # fts_per_rank = np.cumsum(ranking[1])
        # print(len(fts_per_rank), len(scores))
        # print(np.argmax(scores))

        n_fts_remaining = [0] + list(range(nfts_tot - (len(scores) - 2) * step, nfts_tot + step, step))
        # print(n_fts_remaining)

        print(len(scores), len(n_fts_remaining))
        # print(n_fts_remaining)
        n_fts_opt = n_fts_remaining[np.argmax(scores)]
        print("Num fts opt:", n_fts_opt, len(fts_sel))
        print(scores[:5])

        ax.plot(n_fts_remaining, scores, label=md)
        ax.vlines(x=n_fts_opt, ymin=0, ymax=max(scores), linestyles=":", colors="black")

    ax.set_ylabel("F1")
    ax.set_xlabel("Number of features remaining")

    ax.legend()
    plt.show()

    pass

plot_rfe()