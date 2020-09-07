import addressbook_pb2
person = addressbook_pb2.Person()
person.id = 1234
person.name = "John Doe"
person.email = "jdoe@example.com"
phone = person.phones.add()
phone.number = "555-4321"
phone.type = addressbook_pb2.Person.HOME

print(person)

address_book = addressbook_pb2.AddressBook()

address_book.people.append(person)

# Write the new address book back to disk.
f = open('ADDRESSES.txt', "wb")
f.write(address_book.SerializeToString())
f.close()

from google.protobuf import json_format

jj = json_format.MessageToJson(address_book)

print(jj)
f = open('ADDRESSES.json', "wb")
f.write(jj.encode())
f.close()
