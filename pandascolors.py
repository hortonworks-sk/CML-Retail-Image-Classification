
def color_negative_red(v):
    color = 'red' if v < 0.1 else 'black'
    return 'color: %s' % color
 

def highlight_greaterthan_1(s):
  
    print("HHHHH")
    print(s.acc)
  
    if s.acc > 1.0:
        return ['background-color: yellow']*5
    else:
        return ['background-color: white']*5
      
def highlight_greaterthan(s,column):
  
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= 1
    return ['background-color: red' if is_max.any() else '' for v in is_max]
  
      
#pdf.head(10).style.format({"surname": lambda x:x.upper()})\
  #.style.background_gradient(cmap='Blues')
  
  
  def highlight_greaterthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= threshold
    return ['background-color: green' if is_max.any() else '' for v in is_max]

def color_col(column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= threshold
    return ['background-color: green' if is_max.any() else '' for v in is_max]


def colr(x):
  return "color: red"


  
#cm = sns.light_palette("green", as_cmap=True)
  
  
      #.apply(highlight_greaterthan, threshold=0.1, column=['customer_id'], axis=1)
  
 #   .apply(highlight_greaterthan, threshold=0.1, column=['acc'], axis=1)

  
#  .applymap(colr)

#pdf.head(10).style.format({"surname": lambda x:x.lower()}).hide_index().applymap(lambda x: f”color: {‘red’ if isinstance(x,str) else ‘black’}”)
     
      
#  pdf.style.background_gradient(cmap='Blues')
# .applymap(lambda x: f”color: {‘red’ if isinstance(x,str) else ‘black’}”)

  
#pdf.style.apply(highlight_greaterthan_1, 'columns')

#pdf.style.apply(highlight_greaterthan, 'acc')


#pdf.style.applymap(color_negative_red)

#pdf.style.applymap(color_negative_red)

