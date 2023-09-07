#https://www.kaggle.com/code/opamusora/top-2-private-solution?scriptVersionId=139988233
#https://towardsdatascience.com/umap-and-k-means-to-classify-characters-league-of-legends-668a788cb3c1



c__ = ['black', 'lime']

colors_ = [c__[i] for i in train['Class']]
seed = 1
load = False

if load:
    pass
else:
    umap_model = umap.UMAP(n_neighbors=20, random_state=seed)
    embedding = umap_model.fit_transform(X)
    emb_test = umap_model.transform(test)

    with open('/kaggle/working/umap.ckpt', 'wb') as f:
        pickle.dump(umap_model, f)

    kmeans = KMeans(n_clusters=4, random_state=seed)
    kmeans.fit(embedding)

    with open('/kaggle/working/kmeans2.ckpt', 'wb') as f:
        pickle.dump(kmeans, f)

    labels2 = kmeans.predict(embedding)
    labels2_test = kmeans.predict(emb_test)

    kmeans = KMeans(n_clusters=12, random_state=seed)
    kmeans.fit(embedding)

    with open('/kaggle/working/kmeans11.ckpt', 'wb') as f:
        pickle.dump(kmeans, f)

    labels11 = kmeans.predict(embedding)
    labels11_test = kmeans.predict(emb_test)



colors = ["r", "g", "b", "yellow", "pink", "orange", "c", "m", 'aquamarine', 'mediumseagreen','black','dimgray','silver','rosybrown','firebrick','tomato','sandybrown','gold','lawngreen','turquoise','dodgerblue','slateblue','navy','fuchsia','orchid']
labels_color = [colors[l] for l in labels11]

plt.scatter(embedding[:, 0], embedding[:, 1], color = colors_)
