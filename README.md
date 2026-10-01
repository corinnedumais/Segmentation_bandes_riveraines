# Segmentation_bandes_riveraines

Projet final réalisé dans le cadre du cours GLO-4030 – Apprentissage par réseaux de neurones profonds, à l'hiver 2023.

## Description

Ce projet vise à automatiser la caractérisation de l'occupation du sol des bandes riveraines au Québec à partir de données LiDAR. Un modèle de segmentation sémantique U-Net a été développé avec PyTorch afin de classifier les pixels en différentes catégories d'occupation du sol.

Le modèle utilise trois bandes issues des données LiDAR :

Modèle de hauteur de la canopée (MHC)
Modèle numérique de terrain (MNT)
Indice topographique d'humidité (TWI)

<img width="560" height="150" alt="image" src="https://github.com/user-attachments/assets/faeaf57f-63e8-404c-81d9-b24775eab72f" />

Une variante du U-Net intégrant des mécanismes d'attention a également été évaluée, ainsi que différentes stratégies d'augmentation des données.

## Résultats

Le meilleur modèle a obtenu sur l'ensemble de validation :

Coefficient Dice : 82 %
Exactitude : 74 %

Le modèle permet de reconstruire la classification d'une ferme complète à partir de tuiles de 32 × 32 pixels.

<img width="571" height="302" alt="image" src="https://github.com/user-attachments/assets/bf4dc7b6-bfaf-4abe-a95e-60e47ea11104" />

## Technologies

Python · PyTorch · NumPy · LiDAR · Deep Learning · U-Net · Semantic Segmentation

Cours : GLO-4030 – Apprentissage par réseaux de neurones profonds
Université Laval — Hiver 2023
Auteur : Corinne Dumais
