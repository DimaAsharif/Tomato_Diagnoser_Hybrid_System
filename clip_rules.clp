;;;;;;;;;;;;;;;;;;;;
;; define templates 
;;;;;;;;;;;;;;;;;;;;

;; template to store disease knowledge
(deftemplate disease-knowledge
   (slot name) ;; disease name
   (multislot symptoms)) ;; for one or more symptoms

;; template for user symptoms
(deftemplate symptom
   (slot name))

;; template for diagnosis result (output)
(deftemplate disease
   (slot name)
   (slot confidence)) ;; confidence score (CS)


;;;;;;;;;;;;;;;;;;;;;;;;;;
;; knowledge base (facts)
;;;;;;;;;;;;;;;;;;;;;;;;;;
(deffacts known-diseases
   (disease-knowledge (name Septoria-leaf-spot) (symptoms small-dark-spots-on-leaves yellow-leaves dark-spots-or-lesions-on-stem-or-fruit lower-leave-affected))
   (disease-knowledge (name Early-blight) (symptoms small-dark-spots-on-leaves yellow-halos-on-leaves dark-spots-or-lesions-on-stem-or-fruit lower-leave-affected leaves-drop))
   (disease-knowledge (name fusarium-wilt) (symptoms yellow-leaves lower-leave-affected leaves-drop leaves-or-plant-wilts dark/brown-inner-tissue-of-stem))
   (disease-knowledge (name Anthracnose) (symptoms small-dark-spots-on-leaves sunken-spots-on-fruit fruit-rot))
   (disease-knowledge (name Root-knot-nematode) (symptoms yellow-leaves leaves-or-plant-wilts root-galls-or-roots-with-eggs swollen-roots stunned-growth-or-smaller-leaves))
   (disease-knowledge (name late-blight) (symptoms dark-spots-or-lesions-on-stem-or-fruit mold-or-white-spots green-gray-edges-on-leaves greasy-gray-spots-on-fruit water-soaked-spot large-dark-spots-on-leaves))
   (disease-knowledge (name Bacterial-Spot) (symptoms small-dark-spots-on-leaves yellow-halos-on-leaves dark-spots-or-lesions-on-stem-or-fruit sunken-spots-on-fruit holes-on-leave water-soaked-areas-on-leaves))
   (disease-knowledge (name mosaic-virus) (symptoms yellow/light-green-spots-or-mottled-leaves yellow-spots-on-leaves stunned-growth-or-smaller-leaves curled-leaves))
   (disease-knowledge (name Blossom-End-Rot) (symptoms dark-spot-at-the-bottom-of-fruit fruit-rot water-soaked-spot leathery-texture-of-fruit))
   (disease-knowledge (name southern-blight) (symptoms dark-spots-or-lesions-on-stem-or-fruit leaves-drop leaves-or-plant-wilts mold-or-white-spots))
   (disease-knowledge (name Yellow-Leaf-Curl-Virus) (symptoms yellow-leaves yellow-spots-on-leaves stunned-growth-or-smaller-leaves curled-leaves flower-drop))
)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; single generic rule for all cases
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defrule diagnose-disease-generic

   ;; Match any disease definition in knowledge base (?d for disease name and ?symptoms for matched symptoms)
   (disease-knowledge (name ?d) (symptoms $?symptoms))
   =>
   ;; trigger once for every matched disease 
   ;; counts the total number of symotoms for a specific disease
   (bind ?total (length$ ?symptoms))
   ;; counter of user symptoms initialy 0, to calculate CS
   (bind ?matches 0)
   
   ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
   ;; Loop through the required symptoms for this disease
   ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
   (progn$ (?s ?symptoms)
      ;; check if current symptom is equal to one in user's data
      (if (any-factp ((?f symptom)) (eq ?f:name ?s))
         ;; if match is found, increment ?matches by 1
         then (bind ?matches (+ ?matches 1)))
   )
   
   ;;;;;;;;;;;;;;;;;;;;;;;;;;
   ;; Calculate confidence score
   ;;;;;;;;;;;;;;;;;;;;;;;;;;
   ;; at least one matched symptom
   (if (> ?matches 0) then
      ;; divid the nuber of matched symptoms by the total symptoms of that disease
      (bind ?cf (/ ?matches ?total))
      ;; Round and display as a percentage 
      (bind ?cf-rounded (/ (round (* ?cf 100)) 100))
      
      (assert (disease (name ?d) (confidence ?cf-rounded)))
      (printout t "Diagnosis: " ?d " (CF: " ?cf-rounded ")" crlf)
   )
)