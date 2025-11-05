graph [
  directed 0

  node [ id 0 label "0" price 0 ]
  node [ id 1 label "1" price 0 ]
  node [ id 2 label "2" price 0 ]
  node [ id 3 label "3" ]
  node [ id 4 label "4" ]
  node [ id 5 label "5" ]

  # Buyer 4 valuations
  edge [ source 4 target 0 valuation 8 ]
  edge [ source 4 target 1 valuation 7 ]
  edge [ source 4 target 2 valuation 6 ]


  # Buyer 5 valuations (conflict on seller 0 with buyer 4)
  edge [ source 5 target 0 valuation 7 ]
  edge [ source 5 target 1 valuation 5 ]
  edge [ source 5 target 2 valuation 2 ]
 

  # Buyer 3 valuations
  edge [ source 3 target 0 valuation 12 ]
  edge [ source 3 target 1 valuation 4 ]
  edge [ source 3 target 2 valuation 2 ]



]