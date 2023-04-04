from transformers import CLIPProcessor, CLIPModel

class Hockey_classifier:

    def __init__(self, teams_filename, sports_filename):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # NHL teams list
        self.teams = self._load_classes(teams_filename)
        # sports list
        self.sports = self._load_classes(sports_filename)

    def _load_classes(self, filename):
        classes = []
        with open(filename) as file:
            for i in file.readlines():
                classes.append(i.strip())
        return classes

    def _classifier(self, classes, image):
        """
        Internal method to get a prediction from the classifier model. 
        Puts a probability on each classes according to the relation 
        between the class and the image relative to the other classes.

        :classes:   a list of class to assing probabilities to
        :image:     the image to generate predictions of
        :return:    a list of (class, probability) for the image
        """
        inputs = self.processor(text=classes, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()[0] # we can take the softmax to get the label probabilities
        results = zip(classes, probs)
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def _binary_classes(self, one_class):
        """
        Creates a list of the class and its opposite class.
        """
        return [one_class, f"not {one_class}"]

    def _is_it(self, this_class, image, min_confidence=0.5):
        """
        Checks if the image is more related to a class or its oppossite.

        :this_class:    the class to check
        :image:         the image to check
        :return:        False if it's more related to the opposite and the 
                        probability of the class if it's more related to the class
        """
        sports = self._classifier(self._binary_classes(this_class), image)
        if sports[0][0] == this_class and sports[0][1] >= min_confidence:
            return sports[0][1]
        return False
    
    def classify(self, image, min_confidence=0.5):
        """
        Prediction function called to identify a team or not from the image passed.

        :image:             the image to check
        :min_confidence:    the minimum threshold to consider a prediction valid
        :return:            [team, [a list of probabilities tested]]
        """
        activity_score = self.activity_filter(image, min_confidence)
        if not activity_score:
            return [None, []]
        sport_score = self.sports_filter(image, min_confidence)
        if not sport_score:
            return [None, [activity_score]]
        team_result = self.teams_classifier(image, min_confidence)
        if not team_result[1]:
            return [None, [activity_score, sport_score]]
        return [team_result[0], [activity_score, sport_score, team_result[1]]]

    def activity_filter(self, image, min_confidence=0.5):
        """
        Checks if the image is sports related.
        """
        return self._is_it("sports", image, min_confidence)

    def sports_filter(self, image, min_confidence=0.5):
        """
        Checks if the image is hockey related.
        """
        sport = self._classifier(self.sports, image)
        if sport[0][0] == "hockey":
            return self._is_it("hockey", image, min_confidence)
        return False

    def teams_classifier(self, image, min_confidence=0.5):
        """
        Checks if the image can be associated to an NHL team.
        """
        team = self._classifier(self.teams, image)[0][0]
        return [team, self._is_it(team, image, min_confidence)]
