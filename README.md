# Nhl_image_classifier
Service d'identification d'équipe de la NHL utilisé pour complémenter https://github.com/WilliamYn/learning-captioning-model

## Description
Ce service est basé sur le classificateur « Zero-Shot » CLIP de OpenAi. Il permet d'identifier les équipes de la Ligue Nationale de Hockey dans les images.

Ce service est encore très expérimental et n’identifie pas toujours les équipes présentes dans les images. Les tests sur l’ensemble de validation démontrent que le nombre d’images identifiées correctement était autour de 40% et le nombre d’images identifiées incorrectement était de 4%. 

Ce service est prévu d’être étendu à la reconnaissance de logo si les performances des tests sur l’ensemble de validation atteignent le seuil de qualité imposé pour la précision. 

Pour le cas spécifique des équipes de sport, la reconnaissance de logo donnait de façon générale des résultats insatisfaisants, dûs à plusieurs facteurs : la présence de logo des commanditaires qui se faisaient détecter, la clarté des logos des équipes, la présence de texte dans les logos, etc. 

Nous avons donc opté pour une approche différente pour la classification d’équipes de sport. Nous avons passé par le classificateur d’images Zero-Shot Clip. Celui-ci est le même que celui utilisé par le captioning app pour classifier les mots selon l’image. Nos tests ont révélé que celui-ci est capable d’identifier de façon assez précise (88% de précision) les équipes de Hockey lorsqu’elles sont présentes dans les images. Par contre, pour éviter d’étiqueter des images incorrectement, le service effectue plusieurs validations et vérifications avant de passer l’image dans le classificateur, ce qui réduit le pourcentage d’identification véridique d’équipe de Hockey de 88% à 40%. Ces vérifications permettent aussi de réduire le taux de mauvais étiquetage d’images quelconques du modèle de 70% à 4%.

## Déploiement
Il y a une image Docker disponnible sur Docker Hub sous le nom de bawsje/nhl_classifier 
https://hub.docker.com/r/bawsje/nhl_classifier

## Utilisation
Le Docker déploie une application Flask sur la route 5000 du conteneur. 
La route par défaut ("/") du service retourne un message "Hello World" pour permettre de valider si le service est opérationnel.
La route pour accéder a la fonctionnalité de classification est POST <hostname>/generate-hockey-team-label.
Ceci est le format des JSON inputs et outputs de cette fonction:

#### Input JSON
```
{
    "image": "image convertit en base64"
}
```
#### Output JSON
```
{
    "prediction": "nom de l'équipe ou string vide"
}
```
