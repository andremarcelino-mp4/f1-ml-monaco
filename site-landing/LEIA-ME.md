Cópia da landing **sem** a Ferrari desmontada e **sem** as perguntas da banca. A metodologia em texto permanece.

O site em produção é o repo `f1-monaco-ml` (Vercel). Para publicar no seu computador:

```bash
git clone https://github.com/andremarcelino-mp4/f1-monaco-ml.git
cd f1-monaco-ml
git checkout main
# deste repositório f1-ml-monaco, branch cursor/metodologia-teardown-91bf:
cp ../f1-ml-monaco/site-landing/index.html landing/
cp ../f1-ml-monaco/site-landing/css/style.css landing/css/
cp ../f1-ml-monaco/site-landing/js/app.js landing/js/
rm -f landing/js/teardown.js
rm -rf landing/tools/teardown
git add landing && git commit -m "Remove garage teardown and banca FAQ"
git push origin main
```

`CONFIG.GITHUB` aponta para f1-ml-monaco (estudo).
