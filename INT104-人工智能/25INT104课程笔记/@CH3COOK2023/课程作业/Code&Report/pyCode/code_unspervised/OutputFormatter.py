# 不需要更改这个包

import TryBestParameters as P
import pandas as pd

data = {
    "k-means": [P.getKMeansStandardizeBest(), P.getKMeansMinMaxBest(), P.getKMeansNormalizeBest()],
    "GMM": [P.getGMMStandardizeBest(), P.getGMMMinMaxBest(), P.getGMMNormalizeBest()],
    "Hierarchical": [P.getHierarchicalStandardizeBest(), P.getHierarchicalMinMaxBest(),P.getHierarchicalNormalizeBest()]
}
index = ["Standard", "Range", "Normalise"]
df = pd.DataFrame(data, index=index)
print(df)

