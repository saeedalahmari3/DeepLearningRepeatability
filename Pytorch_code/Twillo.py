from twilio.rest import Client

def sendMessage(message):
	# Your Account SID from twilio.com/console
	account_sid = "AC9971e92be8fa45d73665553569e659a4"
	# Your Auth Token from twilio.com/console
	auth_token  = "83e6ac2bc61f369e640664dbd8cfa6dd"

	client = Client(account_sid, auth_token)

	message = client.messages.create(
    	to="+15713038961", 
    	from_="+13019173340",
    	body=message)

