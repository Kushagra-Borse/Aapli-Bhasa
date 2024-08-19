import Bhasha

while True:
    text = input('Aapali Bhasha > ')
    # result, error =  Bhasha.run('<command line>/<stdin>', text)
    result, error =  Bhasha.run('<stdin>', text)
    
    if error:
        print(error.as_string())
    elif result:
        print(result)