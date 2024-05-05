import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:get/get.dart';
import 'dart:convert';

class ChatController extends GetxController {
  final TextEditingController textController = TextEditingController();
  final FocusNode textFocusNode = FocusNode();
  final RxList<String> messages = <String>[].obs;

  Future<void> sendMessage(String message) async {
    messages.add('You: $message');

    final response = await http.post(
      Uri.parse('http://localhost:8000/chat/'),
      headers: <String, String>{
        'Content-Type': 'application/json',
      },
      body: jsonEncode(<String, String>{
        'message': message,
      }),
    );

    if (response.statusCode == 200) {
      Map<String, dynamic> data = jsonDecode(response.body);
      String botResponse = data['response'];
      messages.add('Bot: $botResponse');
    } else {
      throw Exception('Failed to load data');
    }

    // Request focus on the TextField after sending message
    textFocusNode.requestFocus();
  }
}
