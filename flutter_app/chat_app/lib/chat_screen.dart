import 'package:chat_app/chat_controller.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:get/get.dart';

class ChatScreen extends StatelessWidget {
  final ChatController chatController = Get.put(ChatController());

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chat App'),
      ),
      body: Column(
        children: <Widget>[
          Expanded(
            child: Obx(
              () => ListView.builder(
                itemCount: chatController.messages.length,
                itemBuilder: (BuildContext context, int index) {
                  return ListTile(
                    title: Text(chatController.messages[index]),
                  );
                },
              ),
            ),
          ),
          Container(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: <Widget>[
                Expanded(
                  child: RawKeyboardListener(
                    focusNode: FocusNode(), // Ensure this widget has focus
                    onKey: (RawKeyEvent event) {
                      if (event is RawKeyDownEvent &&
                          event.logicalKey == LogicalKeyboardKey.enter) {
                        if (chatController.textController.text.isNotEmpty) {
                          chatController
                              .sendMessage(chatController.textController.text);
                          chatController.textController.clear();
                        }
                      }
                    },
                    child: TextField(
                      focusNode: chatController.textFocusNode,
                      onSubmitted: (val) {
                        chatController
                            .sendMessage(chatController.textController.text);
                        chatController.textController.clear();
                      },
                      controller: chatController.textController,
                      decoration: const InputDecoration(
                        hintText: 'Enter your message...',
                      ),
                    ),
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.send),
                  onPressed: () {
                    if (chatController.textController.text.isNotEmpty) {
                      chatController
                          .sendMessage(chatController.textController.text);
                      chatController.textController.clear();
                    }
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
