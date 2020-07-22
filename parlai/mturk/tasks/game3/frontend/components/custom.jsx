/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from 'react';
import { 
  Form,
  FormControl, 
  Button, 
  Radio,
  Col,
  ControlLabel,
  FormGroup,
  FormLabel } from 'react-bootstrap';
import Slider from 'rc-slider';
import { getCorrectComponent } from "./core_components.jsx";
import $ from 'jquery';

class IdleResponse extends React.Component {
  render() {
    // TODO maybe move to CSS?
    let pane_style = {
      paddingLeft: '25px',
      paddingTop: '20px',
      paddingBottom: '20px',
      paddingRight: '25px',
      float: 'left',
    };

    return (
      <div
        id="response-type-idle"
        className="response-type-module"
        style={pane_style}
      >
        <span>
          Other people on the HIT are still working...
        </span>
      </div>
    );
  }
}

class Hourglass extends React.Component {
  render() {
    // TODO move to CSS document
    let hourglass_style = {
      marginTop: '-1px',
      marginRight: '5px',
      display: 'inline',
      float: 'left',
    };

    // TODO animate?
    return (
      <div id="hourglass" style={hourglass_style}>
        <span className="glyphicon glyphicon-hourglass" aria-hidden="true" />
      </div>
    );
  }
}

class WaitingMessage extends React.Component {
  render() {
    let message_style = {
      float: 'left',
      display: 'table',
      backgroundColor: '#fff',
    };
    let text = 'Waiting for the system or for someone else on the HIT to complete their task...';
    if (this.props.world_state == 'waiting') {
      text = 'Waiting to pair with a task...';
    }
    return (
      <div
        id="waiting-for-message"
        className="row"
        style={{ marginLeft: '0', marginRight: '0' }}
      >
        <div className="alert alert-warning" role="alert" style={message_style}>
          <Hourglass />
          <span style={{ fontSize: '16px' }}>{text}</span>
        </div>
      </div>
    );
  }
}

class WriterResponse extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      entailText: '',
      contradictText: '',
      neutralText: '',
      taskData: [],
      sending: false};
    this.handleInputChange = this.handleInputChange.bind(this);
    // this.handleEnterKey = this.handleEnterKey.bind(this);
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    // Only change in the active status of this component should cause a
    // focus event. Not having this would make the focus occur on every
    // state update (including things like volume changes)
    if (this.props.active && !prevProps.active) {
      $('input#id_text_input').focus();
    }
    this.props.onInputResize();
  }

  tryMessageSend() {
    if (this.state.entailText != '' && this.state.contradictText != '' && this.state.neutralText != '' && this.props.active && !this.state.sending) {
      this.setState({ sending: true });
      this.props.onMessageSend(this.state.entailText, this.state.contradictText , this.state.neutralText, {}, () =>
        this.setState({ entailText: '', contradictText: '', neutralText: '', sending: false })
      );
    }
  }

  handleKeyPress(e) {
    if (e.key === 'Enter') {
      this.tryMessageSend();
      e.stopPropagation();
      e.nativeEvent.stopImmediatePropagation();
    }
  }

  handleInputChange(event) {
    console.log(event)
    let target = event.target;
    let value = target.value;
    let name = target.name;

    this.setState({ [name]: value });
  }

  updateValue(amount) {
    if ((amount != '' && isNaN(amount)) || amount < 0) {
      return;
    }
    amount = amount == '' ? 0 : amount;
    this.setState({ entailText: '' + amount });
  }

  render() {
    // TODO maybe move to CSS?
    let pane_style = {
      paddingLeft: '25px',
      paddingTop: '20px',
      paddingBottom: '20px',
      paddingRight: '25px',
      float: 'left',
      width: '100%',
      height: '300px',
      overflowY: 'scroll',
    };
    let input_style = {
      height: '50px',
      width: '100%',
      display: 'block',
      float: 'left',
    };
    let submit_style = {
      width: '100px',
      height: '100%',
      fontSize: '16px',
      float: 'left',
      marginLeft: '10px',
      padding: '0px',
    };

    let text_entail = (
      <div>
        Please write a claim that is <font color="#E8684C"> Definitely Correct </font> about the situation or event in the prompt,
      </div>
      );
    let entail_input = (
      <div>
        <ControlLabel>{text_entail}</ControlLabel>
        <FormControl
          componentClass="textarea"
          rows="3"
          id="id_text_input0"
          name="entailText"
          style={{
            width: '80%',
            height: '100%',
            float: 'left',
            fontSize: '16px',
          }}
          value={this.state.entailText}
          placeholder="Please enter you claim here..."
          onKeyPress={e => this.handleKeyPress(e)}
          onChange={this.handleInputChange}
          disabled={!this.props.active || this.state.sending}
        />
      </div>
    );

    // onChange={e => this.updateValue(e.target.value)}

    let text_contradict = (
      <div>
        Please write a claim that is <font color="#E8684C"> Definitely Incorrect </font> about the situation or event in the prompt,
      </div>
      );
    let contradict_input = (
      <div>
        <ControlLabel>{text_contradict}</ControlLabel>
        <FormControl
          componentClass="textarea"
          rows="3"
          id="id_text_input1"
          name="contradictText"
          style={{
            width: '80%',
            height: '100%',
            float: 'left',
            fontSize: '16px',
          }}
          value={this.state.contradictText}
          placeholder="Please enter you claim here..."
          onKeyPress={e => this.handleKeyPress(e)}
          onChange={this.handleInputChange}
          disabled={!this.props.active || this.state.sending}
        />
      </div>
    );

    let text_neutral = (
      <div>
        Please write a claim that is <font color="#E8684C"> Neither </font> definitely correct nor definitely incorrect about the situation or event in the prompt,
      </div>
      );
    let neutral_input = (
      <div>
        <ControlLabel>{text_neutral}</ControlLabel>
        <FormControl
          componentClass="textarea"
          rows="3"
          id="id_text_input2"
          name="neutralText"
          style={{
            width: '80%',
            height: '100%',
            float: 'left',
            fontSize: '16px',
          }}
          value={this.state.neutralText}
          placeholder="Please enter you claim here..."
          onKeyPress={e => this.handleKeyPress(e)}
          onChange={this.handleInputChange}
          disabled={!this.props.active || this.state.sending}
        />
      </div>
    );

    // TODO attach send message callback
    let submit_button = (
      <Button
        className="btn btn-primary"
        style={submit_style}
        id="id_send_msg_button"
        disabled={
          this.state.entailText == '' || this.state.contradictText == '' || this.state.neutralText == '' || !this.props.active || this.state.sending
        }
        onClick={() => this.tryMessageSend()}
      >
        Submit
      </Button>
    );

    return (
      <div
        id="response-type-text-input"
        className="response-type-module"
        style={pane_style}
      >
        <div style={input_style}>
        {entail_input}
        {contradict_input}
        {neutral_input}
        {submit_button}
        </div>
      </div>
    );
  }
}

class EvaluatorResponse extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      textreason: '',
      claimChoice: '',
      validation1: '',
      validation2: '',
      taskData: [],
      sending: false};
    this.handleInputChange = this.handleInputChange.bind(this);
    // this.handleEnterKey = this.handleEnterKey.bind(this);
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    // Only change in the active status of this component should cause a
    // focus event. Not having this would make the focus occur on every
    // state update (including things like volume changes)
    if (this.props.active && !prevProps.active) {
      $('input#id_text_input').focus();
    }
    this.props.onInputResize();
  }

  tryMessageSend() {
    if (this.props.active && !this.state.sending) {
      this.setState({ sending: true });
      this.props.onMessageSend(this.state.claimChoice, this.state.textreason, {}, {}, () =>
        this.setState({ claimChoice: '', textreason: '', sending: false })
      );
    }
  }

  handleInputChange(event) {
    console.log(event)
    let target = event.target;
    let value = target.value;
    let name = target.name;

    this.setState({ [name]: value });
  }

  checkValidData() {
    console.log(this.state);
    if (this.state.validation1 !== "") {
      let response_data = {
        validation1: this.state.validation1,
        entailText: this.state.entailText
      };
      this.props.onValidDataChange(true, response_data);
      return;
    }
    this.props.onValidDataChange(false, {});
  }


  handleKeyPress(e) {
    if (e.key === 'Enter') {
      this.tryMessageSend();
      e.stopPropagation();
      e.nativeEvent.stopImmediatePropagation();
    }
  }

  _showMessage = (bool) => {
    this.setState({
      showMessage: bool
    });
  }

  updateValue(amount) {
    if ((amount != '' && isNaN(amount)) || amount < 0) {
      return;
    }
    amount = amount == '' ? 0 : amount;
    this.setState({ entailText: '' + amount });
  }

  render() {
    // TODO maybe move to CSS?
    let pane_style = {
      paddingLeft: '25px',
      paddingTop: '20px',
      paddingBottom: '20px',
      paddingRight: '25px',
      float: 'left',
      width: '100%',
      height: '300px',
      overflowY: 'scroll',
    };
    //  TODO: make height variable, a percentage?
    let input_style = {
      height: '50px',
      width: '100%',
      display: 'block',
      float: 'left',

    };
    let input_inline_style = {
      height: '50px',
      width: '100%',
      display: 'inline-block',
      float: 'left',
    };
    let submit_style = {
      width: '100px',
      height: '100%',
      fontSize: '16px',
      float: 'left',
      marginLeft: '10px',
      padding: '0px',
    };

    const approvebox  = {
      backgroundColor: '#66B46E',
      borderRadius: 3,
      padding: '5px',
      border: '1px solid #44864B',
      color: 'white',
      width: 'fit-content',
    }

    const rejectbox  = {
      backgroundColor: '#E8684C',
      borderRadius: 3,
      padding: '5px',
      border: '1px solid #D34000',
      color: 'white',
    }

    const claimbox  = {
      backgroundColor: '#636363',
      borderRadius: 3,
      padding: '5px',
      border: '1px solid #393939',
      color: 'white',
    }

    const inline_text = {
      display: "inline-block",
    };

    const approve = "Valid"
    const invalid = "Invalid"
    const text_claim1 = "Your claim"
    const text_claim2 = "Their claim"
    let validation_buttons_1 = (
      <div>
        <Form
          horizontal
          style={{ backgroundColor: "#eeeeee", paddingBottom: "10px", 
            display: "inline-block", verticalAlign:"top" }}
        >
          <div className="container" style={{ width: "auto" }}>
          <ControlLabel style={{paddingRight: "30px"}}> {text_claim1}, </ControlLabel>
            <FormGroup 
            style={{display: "inline-block", verticalAlign: "top"}}
            >
              <Col sm={6}>
                <Radio
                  name="validation1"
                  value={approve}
                  style={{ width: "100%" }}
                  checked={this.state.validation1 == approve}
                  onChange={this.handleInputChange}
                >
                  <div style={approvebox}> Approve </div>
                </Radio>
              </Col>
              <Col sm={6}>
                <Radio
                  name="validation1"
                  value={invalid}
                  style={{ width: "100%" }}
                  checked={this.state.validation1 == invalid}
                  onChange={this.handleInputChange}
                >
                  <div style={rejectbox}> Invalid </div>
                </Radio>
              </Col>
            </FormGroup>
          </div>
        </Form>
      </div>
    );

    let validation_buttons_2 = (
      <div>
        <Form
          horizontal
          style={{ backgroundColor: "#eeeeee", paddingBottom: "10px",
            display: "inline-block", verticalAlign:"top" }}
        >
          <div className="container" style={{ width: "auto" }}>
          <ControlLabel style={{paddingRight: "30px"}}> {text_claim2}, </ControlLabel>
            <FormGroup
            style={{display: "inline-block", verticalAlign:"top"}}
            >
              <Col sm={6}>
                <Radio
                  name="validation2"
                  value={approve}
                  style={{ width: "100%" }}
                  checked={this.state.validation2 == approve}
                  onChange={this.handleInputChange}
                  onClick={this._showMessage.bind(null, true)}
                >
                  <div style={approvebox}> Approve </div>
                </Radio>
              </Col>
              <Col sm={6}>
                <Radio
                  name="validation2"
                  value={invalid}
                  style={{ width: "100%" }}
                  checked={this.state.validation2 == invalid}
                  onChange={this.handleInputChange}
                  onClick={this._showMessage.bind(null, true)}
                >
                  <div style={rejectbox}> Invalid </div>
                </Radio>
              </Col>
            </FormGroup>
          </div>
        </Form>
      </div>
    );

    const text_rank1 = "Which claim do you think is better?" ;
    let rank1 = (
      <div>
        <Form
          horizontal
          style={{ backgroundColor: "#eeeeee", paddingBottom: "10px",
            display: "inline-block", verticalAlign:"top" }}
        >
          <div className="container" style={{ width: "auto" }}>
            <ControlLabel  style={{paddingRight: "30px"}}> {text_rank1} </ControlLabel>
            <FormGroup
            style={{display: "inline-block", verticalAlign:"top"}}
            >
              <Col sm={6}>
                <Radio
                  name="claimChoice"
                  value={text_claim1}
                  style={{ width: "100%" }}
                  checked={this.state.claimChoice == text_claim1}
                  onChange={this.handleInputChange}
                >
                  <div style={claimbox}> Your claim </div>
                </Radio>
              </Col>
              <Col sm={6}>
                <Radio
                  name="claimChoice"
                  value={text_claim2}
                  style={{ width: "100%" }}
                  checked={this.state.claimChoice == text_claim2}
                  onChange={this.handleInputChange}
                >
                  <div style={claimbox}> Their claim </div>
                </Radio>
              </Col>
            </FormGroup>
          </div>
        </Form>
      </div>
    );

    let text_reasoning = (
      <div>
        Optionally (but encouraged), please provide a brief justification for your ranking
      </div>
      );
    let text_reason = (
      <div>
        <ControlLabel>{text_reasoning}</ControlLabel>
        <FormControl
          componentClass="textarea"
          rows="1"
          id="id_textreason"
          name="textreason"
          style={{
            width: '80%',
            height: '100%',
            float: 'left',
            fontSize: '16px',
          }}
          value={this.state.textreason}
          placeholder="Optionally add your justification here..."
          onChange={this.handleInputChange}
          disabled={!this.props.active || this.state.sending}
        />
      </div>
    );
    // onKeyPress={e => this.handleKeyPress(e)}
    
    let submit_button = (
      <Button
        className="btn btn-primary"
        style={submit_style}
        id="id_send_msg_button"
        disabled={
          this.state.claimChoice == '' || !this.props.active || this.state.sending
        }
        onClick={() => this.tryMessageSend()}
      >
        Submit
      </Button>
    );

    return (
      <div
        id="response-type-text-input"
        className="response-type-module"
        style={pane_style}
      >
        <div style={input_style}>
        {rank1}
        {text_reason}
        {submit_button}
        </div>
      </div>
    );
  }
}

class FeedbackMode extends React.Component {
  render() {
    // TODO maybe move to CSS?
    let pane_style = {
      paddingLeft: '25px',
      paddingTop: '20px',
      paddingBottom: '20px',
      paddingRight: '25px',
      float: 'left',
    };
    return (
      <div
        id="response-type-done"
        className="response-type-module"
        style={pane_style}
      >
        <i>Feedback!</i>
      </div>
    );
  }
}

class OnboardResponse extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      // textreason: '',
      // claimChoice: '',
      onboardvalidation: '',
      // validation2: '',
      taskData: [],
      sending: false};
    this.handleInputChange = this.handleInputChange.bind(this);
    // this.handleEnterKey = this.handleEnterKey.bind(this);
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    // Only change in the active status of this component should cause a
    // focus event. Not having this would make the focus occur on every
    // state update (including things like volume changes)
    if (this.props.active && !prevProps.active) {
      $('input#id_text_input').focus();
    }
    this.props.onInputResize();
  }

  tryMessageSend() {
    if (this.state.entailText != '' && this.state.contradictText != '' && this.state.neutralText != '' && this.props.active && !this.state.sending) {
      this.setState({ sending: true });
      this.props.onMessageSend(this.state.entailText, this.state.contradictText , this.state.neutralText, {}, () =>
        this.setState({ entailText: '', contradictText: '', neutralText: '', sending: false })
      );
    }
  }

  tryMessageSend() {
    if (this.props.active && !this.state.sending) {
      this.setState({ sending: true });
      this.props.onMessageSend(this.state.onboardvalidation, {}, {}, {}, () =>
        this.setState({ onboardvalidation: '', sending: false })
      );
    }
  }

  handleInputChange(event) {
    console.log(event)
    let target = event.target;
    let value = target.value;
    let name = target.name;

    this.setState({ [name]: value });
  }

  checkValidData() {
    console.log(this.state);
    if (this.state.validation1 !== "") {
      let response_data = {
        validation1: this.state.validation1,
        entailText: this.state.entailText
      };
      this.props.onValidDataChange(true, response_data);
      return;
    }
    this.props.onValidDataChange(false, {});
  }


  handleKeyPress(e) {
    if (e.key === 'Enter') {
      this.tryMessageSend();
      e.stopPropagation();
      e.nativeEvent.stopImmediatePropagation();
    }
  }

  _showMessage = (bool) => {
    this.setState({
      showMessage: bool
    });
  }

  updateValue(amount) {
    if ((amount != '' && isNaN(amount)) || amount < 0) {
      return;
    }
    amount = amount == '' ? 0 : amount;
    this.setState({ entailText: '' + amount });
  }

  render() {
    // TODO maybe move to CSS?
    let pane_style = {
      paddingLeft: '25px',
      paddingTop: '20px',
      paddingBottom: '20px',
      paddingRight: '25px',
      float: 'left',
      width: '100%',
      height: '150px',
      overflowY: 'scroll',
    };

    let input_style = {
      height: '50px',
      width: '100%',
      display: 'block',
      float: 'left',

    };
    let input_inline_style = {
      height: '50px',
      width: '100%',
      display: 'inline-block',
      float: 'left',
    };
    let submit_style = {
      width: '100px',
      height: '100%',
      fontSize: '16px',
      float: 'left',
      marginLeft: '10px',
      padding: '0px',
    };

    const approvebox  = {
      backgroundColor: '#66B46E',
      borderRadius: 3,
      padding: '5px',
      border: '1px solid #44864B',
      color: 'white',
      width: 'fit-content',
    }

    const rejectbox  = {
      backgroundColor: '#E8684C',
      borderRadius: 3,
      padding: '5px',
      border: '1px solid #D34000',
      color: 'white',
    }

    const inline_text = {
      display: "inline-block",
    };

    const onboard_text = "Given the prompt sentence and the label, the the claim valid?" ;
    const approve = "Valid"
    const invalid = "Invalid"
    let validation_buttons = (
      <div>
        <Form
          horizontal
          style={{ backgroundColor: "#eeeeee", paddingBottom: "10px", 
            display: "inline-block", verticalAlign:"top" }}
        >
          <div className="container" style={{ width: "auto" }}>
            <ControlLabel  style={{paddingRight: "30px"}}> {onboard_text} </ControlLabel>
            <FormGroup 
            style={{display: "inline-block", verticalAlign: "top"}}
            >
              <Col sm={6}>
                <Radio
                  name="onboardvalidation"
                  value={approve}
                  style={{ width: "100%" }}
                  checked={this.state.onboardvalidation == approve}
                  onChange={this.handleInputChange}
                >
                  <div style={approvebox}> Approve </div>
                </Radio>
              </Col>
              <Col sm={6}>
                <Radio
                  name="onboardvalidation"
                  value={invalid}
                  style={{ width: "100%" }}
                  checked={this.state.onboardvalidation == invalid}
                  onChange={this.handleInputChange}
                >
                  <div style={rejectbox}> Invalid </div>
                </Radio>
              </Col>
            </FormGroup>
          </div>
        </Form>
      </div>
    );
    
    let submit_button = (
      <Button
        className="btn btn-primary"
        style={submit_style}
        id="id_send_msg_button"
        disabled={
          this.state.onboardvalidation == '' || !this.props.active || this.state.sending
        }
        onClick={() => this.tryMessageSend()}
      >
        Submit
      </Button>
    );

    return (
      <div
        id="response-type-text-input"
        className="response-type-module"
        style={pane_style}
      >
        <div style={input_style}>
        {validation_buttons}
        {submit_button}
        </div>
      </div>
    );
  }
}

class ResponsePane extends React.Component {
  render() {
    let v_id = this.props.v_id;
    let XDoneResponse = getCorrectComponent("XDoneResponse", v_id);
    let XFormResponse = getCorrectComponent('XFormResponse', v_id);

    let response_pane = null;
    switch (this.props.chat_state) {
      case 'done':
      case 'inactive':
        response_pane = <XDoneResponse {...this.props} />;
        break;
      case 'text_input':
      case 'waiting':
        if (this.props.task_data && this.props.task_data['respond_with_form'] && this.props.task_data['writing'] !== true && this.props.task_data['feedback'] !== true && this.props.task_data['onboard'] !== true ) {
          response_pane = (
            <EvaluatorResponse
              {...this.props}
              active={this.props.chat_state == 'text_input'}
            />
          );
        } else if (this.props.task_data['writing'] === true && this.props.task_data['feedback'] !== true){
          response_pane = (
            <WriterResponse
              {...this.props}
              active={this.props.chat_state == 'text_input'}
            />
          );
        } else if (this.props.task_data['feedback'] === true){
          response_pane = <FeedbackMode {...this.props} />;
        } else if (this.props.task_data && this.props.task_data['respond_with_form'] && this.props.task_data['onboard'] === true) {
          response_pane = (
            <OnboardResponse
              {...this.props}
              active={this.props.chat_state == 'text_input'}
            />
          );
        } else {
          response_pane = <XDoneResponse {...this.props} />;
        }
        break;
      case 'idle':
      default:
        response_pane = <IdleResponse {...this.props} />;
        break;
    }

    return (
      <div
        id="right-bottom-pane"
        style={{ width: "100%", backgroundColor: "#eee" }}
      >
        {response_pane}
      </div>
    );
  }
}

class MessageList extends React.Component {
  makeMessages() {
    let agent_id = this.props.agent_id;
    let messages = this.props.messages;
    // Handles rendering messages from both the user and anyone else
    // on the thread - agent_ids for the sender of a message exist in
    // the m.id field.
    
    let onClickMessage = this.props.onClickMessage;
    if (typeof onClickMessage !== 'function') {
      onClickMessage = idx => {};
    }
    return messages.map((m, idx) => (
      <div key={m.message_id} onClick={() => onClickMessage(idx)}>
        <CoreChatMessage
          is_self={m.id == agent_id}
          agent_id={m.id}
          message={m.text}
          task_data={m.task_data}
          task_data2={m.task_data2}
          task_data3={m.task_data3}
          message_id={m.message_id}
          duration={this.props.is_review ? m.duration : undefined}
        />
      </div>
    ));
  }

  render() {
    return (
      <div id="message_thread" style={{ width: '100%' }}>
        {this.makeMessages()}
      </div>
    );
  }
}

class CoreChatMessage extends React.Component {
  render() {
    let float_loc = 'left';
    let alert_class = 'alert-warning';
    if (this.props.is_self) {
      float_loc = 'right';
      alert_class = 'alert-info';
    }
    let duration = null;
    if (this.props.duration !== undefined) {
      let duration_seconds = Math.floor(this.props.duration / 1000) % 60;
      let duration_minutes = Math.floor(this.props.duration / 60000);
      let min_text = duration_minutes > 0 ? duration_minutes + ' min' : '';
      let sec_text = duration_seconds > 0 ? duration_seconds + ' sec' : '';
      duration = (
        <small>
          <br />
          <i>Duration: </i>
          {min_text + ' ' + sec_text}
        </small>
      );
    }

    let agent_id = this.props.agent_id;
    let display_message = null;
    if (this.props.task_data && !this.props.task_data['writing'] && !this.props.task_data['ranking'] && !this.props.task_data['feedback'] && !this.props.task_data['onboard'] && !this.props.task_data['respond_with_form'] && 
      (Object.entries(this.props.task_data3).length === 0)) {
      let hypothesis_entailment = this.props.message;
      let hypothesis_contradiction = this.props.task_data;
      let hypothesis_neutral = this.props.task_data2;
      display_message = (
        '\n<u>Def. correct</u>: ' + hypothesis_entailment + '\n' + 
        '<u>Def incorrect</u>: ' + hypothesis_contradiction + '\n' +
        '<u>Neither</u>: ' + hypothesis_neutral
      );
    } else {
      display_message = this.props.message;
    }
    return (
      <div className={'row'} style={{ marginLeft: '0', marginRight: '0' }}>
        <div
          className={'alert ' + alert_class}
          role="alert"
          style={{ float: float_loc, display: 'table' }}
        >
          <span style={{ fontSize: '16px', whiteSpace: 'pre-wrap' }}>
            <span dangerouslySetInnerHTML={{ __html: agent_id }} />: <span dangerouslySetInnerHTML={{ __html: display_message }} />
          </span>
          {duration}
        </div>
      </div>
    );
  }
}

export default {
  // ComponentName: CustomReplacementComponentMap
  XWaitingMessage: { default: WaitingMessage },
  XResponsePane: { default: ResponsePane },
  XMessageList: { default: MessageList },
  XChatMessage: { default: CoreChatMessage },
};
