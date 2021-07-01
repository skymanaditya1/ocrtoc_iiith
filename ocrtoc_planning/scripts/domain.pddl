(define (domain discrete-tamp)
  (:requirements :strips :equality)
  (:constants q100)
  (:predicates
    (Conf ?q)
    (Block ?b)
    (Pose ?p)
    (Kin ?q ?p)
    (AtPose ?p ?q)
    (AtConf ?q)
    (Holding ?b)
    (HandEmpty)
    (CFree ?p1 ?p2)
    (Collision ?p1 ?p2)
    (Unsafe ?p)
    (CanMove)
    (Ontop ?q)
  )
  (:functions
    (Distance ?q1 ?q2)
  )
  (:action move
    :parameters (?q1 ?q2)
    :precondition (and (Conf ?q1) (Conf ?q2)
                       (AtConf ?q1) (CanMove) (Ontop ?q2))
    :effect (and (AtConf ?q2)
                 (not (AtConf ?q1)) (not (CanMove))
                 (increase (total-cost) (Distance ?q1 ?q2)))
  )
  (:action pick
    :parameters (?b ?p ?q)
    :precondition (and (Block ?b) (Kin ?q ?p)
                       (AtConf ?q) (AtPose ?b ?p) (HandEmpty))
    :effect (and (Holding ?b) (CanMove)
                 (not (AtPose ?b ?p)) (not (HandEmpty))
                 (increase (total-cost) 1))
  )
  (:action place
    :parameters (?b ?p ?q)
    :precondition (and (Block ?b) (Kin ?q ?p)
                       (AtConf ?q) (Holding ?b) (not (Unsafe ?p)))
    :effect (and (AtPose ?b ?p) (HandEmpty) (CanMove)
                 (not (Holding ?b))
                 (increase (total-cost) 1))
  )
  (:derived (Unsafe ?p)
    (exists (?b2 ?p2) (and (Pose ?p) (Block ?b2) (Pose ?p2)
                          (not (CFree ?p ?p2))
                          (AtPose ?b2 ?p2)))
  )
)